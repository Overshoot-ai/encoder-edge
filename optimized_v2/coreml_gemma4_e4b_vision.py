"""Isolated fixed-shape Core ML experiment for the Gemma 4 E4B vision encoder.

The Core ML input is the processor's fixed, unpadded patch tensor. Position IDs,
RoPE values, and the spatial pooling layout are constants in the exported graph.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file


class StableRMSNorm(torch.nn.Module):
    """RMSNorm with bounded intermediates for Core ML's FP16 lowering."""

    def __init__(self, source: torch.nn.Module):
        super().__init__()
        self.eps = source.eps
        self.with_scale = source.with_scale
        if self.with_scale:
            self.weight = source.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scale = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        scaled = hidden_states / scale
        mean_squared = scaled.pow(2).mean(dim=-1, keepdim=True)
        mean_squared = mean_squared + self.eps / scale.pow(2)
        output = scaled * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            output = output * self.weight
        return output


def replace_rms_norms(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Gemma4RMSNorm":
            setattr(module, name, StableRMSNorm(child))
        else:
            replace_rms_norms(child)


def disable_broken_system_torchvision() -> None:
    """Force the PIL processor when a system torchvision mismatches PyTorch."""
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils.is_torchvision_available = lambda: False
    transformers_utils.is_torchvision_available = lambda: False


class FixedGemma4VisionEncoder(torch.nn.Module):
    """Traceable Gemma 4 tower with shape-dependent work precomputed."""

    def __init__(self, tower, projector, position_ids: torch.Tensor, layers: int):
        super().__init__()
        self.input_proj = tower.patch_embedder.input_proj
        self.layers = torch.nn.ModuleList(list(tower.encoder.layers[:layers]))
        self.projector = projector
        self.hidden_size = tower.config.hidden_size
        self.pooling_kernel_size = tower.config.pooling_kernel_size

        position_ids = position_ids.to(torch.long)
        grid_width = int(position_ids[0, :, 0].max().item()) + 1
        grid_height = int(position_ids[0, :, 1].max().item()) + 1
        if grid_width * grid_height != position_ids.shape[1]:
            raise ValueError("Fixed export requires one unpadded rectangular patch grid")
        if grid_width % self.pooling_kernel_size or grid_height % self.pooling_kernel_size:
            raise ValueError("Patch grid must be divisible by the pooling kernel")
        self.grid_width = grid_width
        self.grid_height = grid_height

        with torch.no_grad():
            table = tower.patch_embedder.position_embedding_table
            fixed_position_embedding = (
                table[0, position_ids[..., 0]] + table[1, position_ids[..., 1]]
            )
            dummy = torch.empty(
                1, position_ids.shape[1], self.hidden_size, dtype=table.dtype
            )
            cos, sin = tower.encoder.rotary_emb(dummy, position_ids)
        self.register_buffer("fixed_position_embedding", fixed_position_embedding)
        self.register_buffer("fixed_cos", cos)
        self.register_buffer("fixed_sin", sin)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_proj(2.0 * (pixel_values - 0.5))
        hidden_states = hidden_states + self.fixed_position_embedding
        position_embeddings = (self.fixed_cos, self.fixed_sin)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                # Only the final dimension (the number of axes) is consulted.
                position_ids=self.fixed_position_embedding[..., :2],
            )

        batch, _, channels = hidden_states.shape
        kernel = self.pooling_kernel_size
        # Two rank-5 reductions are equivalent to a rank-6 2-D pool and remain
        # within Core ML's maximum tensor rank.
        hidden_states = hidden_states.reshape(
            batch, self.grid_height, self.grid_width // kernel, kernel, channels
        ).mean(dim=3)
        hidden_states = hidden_states.reshape(
            batch,
            self.grid_height // kernel,
            kernel,
            self.grid_width // kernel,
            channels,
        ).mean(dim=2)
        hidden_states = hidden_states.reshape(batch, -1, channels)
        hidden_states = hidden_states.float() * math.sqrt(self.hidden_size)
        if getattr(self, "std_bias", None) is not None:
            hidden_states = (hidden_states - self.std_bias.float()) * self.std_scale.float()
        return self.projector(hidden_states.to(pixel_values.dtype))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/gemma-4-e4b/client")
    )
    parser.add_argument(
        "--image", type=Path, default=Path("artifacts/gemma-4-12b/sample.png")
    )
    parser.add_argument("--output", type=Path, default=Path("coreml-gemma4-e4b"))
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-palettize", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--extract-tower-from",
        type=Path,
        help="Cut the projector from an existing converted .mlpackage",
    )
    return parser.parse_args()


def load_experiment(args: argparse.Namespace, stable_norms: bool = True):
    disable_broken_system_torchvision()
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        Gemma4ImageProcessorPil,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )

    config = Gemma4Config.from_pretrained(args.artifact)
    processor = Gemma4ImageProcessorPil.from_pretrained(args.artifact)
    tower = Gemma4VisionModel(config.vision_config).eval()
    projector = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).eval()
    state = load_file(args.artifact / "vision.safetensors")
    tower.load_state_dict(
        {key.removeprefix("vision_tower."): value for key, value in state.items() if key.startswith("vision_tower.")}
    )
    projector.load_state_dict(
        {key.removeprefix("embed_vision."): value for key, value in state.items() if key.startswith("embed_vision.")}
    )
    tower.float()
    projector.float()
    if stable_norms:
        replace_rms_norms(tower)
        replace_rms_norms(projector)

    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (args.width, args.height),
        method=Image.Resampling.LANCZOS,
    )
    inputs = processor(images=image, return_tensors="pt")
    padded_pixels = inputs["pixel_values"].float()
    padded_positions = inputs["image_position_ids"].long()
    valid = ~(padded_positions == -1).all(dim=-1)
    valid_count = int(valid.sum().item())
    pixels = padded_pixels[:, :valid_count].contiguous()
    positions = padded_positions[:, :valid_count].contiguous()
    layers = args.layers or config.vision_config.num_hidden_layers
    if not 1 <= layers <= config.vision_config.num_hidden_layers:
        raise ValueError(f"--layers must be in [1, {config.vision_config.num_hidden_layers}]")
    wrapper = FixedGemma4VisionEncoder(tower, projector, positions, layers).eval()
    return wrapper, pixels, positions, padded_pixels, padded_positions, tower, projector


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float32) - reference.astype(np.float32)
    reference_f32 = reference.astype(np.float32)
    return {
        "reference_min": float(np.min(reference_f32)),
        "reference_max": float(np.max(reference_f32)),
        "candidate_min": float(np.min(candidate)),
        "candidate_max": float(np.max(candidate)),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference_f32)),
        "cosine_similarity": float(
            np.dot(reference_f32.ravel(), candidate.astype(np.float32).ravel())
            / (np.linalg.norm(reference_f32) * np.linalg.norm(candidate.astype(np.float32)))
        ),
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def extract_tower_model(projected_path: Path, tower_path: Path) -> None:
    import coremltools as ct

    source = ct.models.MLModel(str(projected_path), skip_model_load=True)
    spec = source.get_spec()
    function = spec.mlProgram.functions["main"]
    block = next(iter(function.block_specializations.values()))
    operations = block.operations

    projector_index, projector_linear = next(
        (index, operation)
        for index, operation in reversed(list(enumerate(operations)))
        if operation.type == "linear"
        and any(
            argument.name.startswith("self_projector_embedding_projection_weight")
            for argument in operation.inputs["weight"].arguments
        )
    )
    projector_output = projector_linear.inputs["x"].arguments[0].name
    needed = {projector_output}
    projector_norm_operations = []
    for operation in reversed(operations[:projector_index]):
        output_names = {output.name for output in operation.outputs}
        if not output_names.intersection(needed):
            continue
        projector_norm_operations.append(operation)
        needed.update(
            argument.name
            for value in operation.inputs.values()
            for argument in value.arguments
        )

    norm_abs = next(
        operation
        for operation in projector_norm_operations
        if operation.type == "abs"
    )
    tower_output = norm_abs.inputs["x"].arguments[0].name
    producer_index, producer = next(
        (index, operation)
        for index, operation in enumerate(operations)
        if any(output.name == tower_output for output in operation.outputs)
    )
    producer_output = next(output for output in producer.outputs if output.name == tower_output)
    producer_output.name = "image_features"
    del operations[producer_index + 1 :]
    block.outputs[0] = "image_features"

    output_type = spec.description.output[0].type.multiArrayType
    del output_type.shape[:]
    output_type.shape.extend([1, 264, 768])
    weights_dir = projected_path / "Data/com.apple.CoreML/weights"
    tower = ct.models.MLModel(spec, weights_dir=str(weights_dir))
    tower.short_description = "Fixed-shape Gemma 4 E4B pre-projector vision tower"
    tower.save(str(tower_path))


def load_fixed_pixels_without_artifact(args: argparse.Namespace) -> torch.Tensor:
    disable_broken_system_torchvision()
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        Gemma4ImageProcessorPil,
    )

    processor = Gemma4ImageProcessorPil()
    image = ImageOps.fit(
        Image.open(args.image).convert("RGB"),
        (args.width, args.height),
        method=Image.Resampling.LANCZOS,
    )
    inputs = processor(images=image, return_tensors="pt")
    positions = inputs["image_position_ids"].long()
    valid_count = int((~(positions == -1).all(dim=-1)).sum().item())
    return inputs["pixel_values"][:, :valid_count].contiguous().float()


def run_extracted_tower_experiment(args: argparse.Namespace) -> None:
    import coremltools as ct

    args.output.mkdir(parents=True, exist_ok=True)
    fp16_path = args.output / "gemma4_e4b_tower_fp16.mlpackage"
    int4_path = args.output / "gemma4_e4b_tower_int4.mlpackage"
    extract_tower_model(args.extract_tower_from, fp16_path)
    fp16_model = ct.models.MLModel(str(fp16_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    if not args.skip_palettize:
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=4, weight_threshold=1024
            )
        )
        int4_model = ct.optimize.coreml.palettize_weights(fp16_model, config=config)
        int4_model.short_description = "Fixed-shape Gemma 4 E4B pre-projector tower (4-bit)"
        int4_model.save(str(int4_path))

    pixels = load_fixed_pixels_without_artifact(args)
    model_input = {"pixel_values": pixels.numpy().astype(np.float16)}
    reference = fp16_model.predict(model_input)["image_features"]
    benchmarks = {}
    if not args.skip_benchmark:
        for name, path in {"fp16": fp16_path, "int4": int4_path}.items():
            if path.exists():
                benchmarks[name] = benchmark_coreml(
                    path, pixels, reference, args.warmup, args.rounds
                )
    result = {
        "source_projected_model": str(args.extract_tower_from),
        "fixed_patch_shape": list(pixels.shape),
        "reference_output_shape": list(reference.shape),
        "fp16_bytes": directory_size(fp16_path),
        "int4_bytes": directory_size(int4_path) if int4_path.exists() else None,
        "benchmarks_cpu_and_ne": benchmarks,
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def summarize_latencies(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": ordered[max(0, math.ceil(len(ordered) * 0.9) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def convert_models(wrapper, pixels, output: Path, palettize: bool) -> dict[str, object]:
    try:
        import coremltools as ct
    except ImportError as error:
        raise SystemExit("Install the experiment dependency with: uv pip install coremltools") from error

    output.mkdir(parents=True, exist_ok=True)
    traced_path = output / "gemma4_e4b_fixed_trace.pt"
    fp16_path = output / "gemma4_e4b_vision_fp16.mlpackage"
    int4_path = output / "gemma4_e4b_vision_int4.mlpackage"
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, pixels, strict=True)
        traced = torch.jit.freeze(traced)
        traced.save(str(traced_path))

    started = time.perf_counter()
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=tuple(pixels.shape), dtype=np.float16)],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    model.short_description = "Fixed-shape Gemma 4 E4B vision encoder (FP16)"
    model.save(fp16_path)
    result: dict[str, object] = {
        "trace_bytes": traced_path.stat().st_size,
        "fp16_bytes": directory_size(fp16_path),
        "fp16_conversion_seconds": time.perf_counter() - started,
    }
    if palettize:
        started = time.perf_counter()
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=4, weight_threshold=1024
            )
        )
        int4_model = ct.optimize.coreml.palettize_weights(model, config=config)
        int4_model.short_description = "Fixed-shape Gemma 4 E4B vision encoder (4-bit weights)"
        int4_model.save(int4_path)
        result.update(
            {
                "int4_bytes": directory_size(int4_path),
                "int4_palettization_seconds": time.perf_counter() - started,
            }
        )
    return result


def benchmark_coreml(
    model_path: Path,
    pixels: torch.Tensor,
    reference: np.ndarray,
    warmup: int,
    rounds: int,
) -> dict[str, object]:
    import coremltools as ct

    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    model_input = {"pixel_values": pixels.numpy().astype(np.float16)}
    for _ in range(warmup):
        model.predict(model_input)
    latencies = []
    output = None
    for _ in range(rounds):
        started = time.perf_counter()
        output = model.predict(model_input)["image_features"]
        latencies.append((time.perf_counter() - started) * 1000)
    if output is None:
        raise RuntimeError("Benchmark produced no output")
    return {
        "latency": summarize_latencies(latencies),
        "error": error_metrics(reference, output),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    }


def main() -> None:
    args = parse_args()
    if args.extract_tower_from is not None:
        run_extracted_tower_experiment(args)
        return
    (
        wrapper,
        pixels,
        positions,
        padded_pixels,
        padded_positions,
        tower,
        projector,
    ) = load_experiment(args)
    args.output.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        fixed_reference = wrapper(pixels).numpy()
        source_error = None
        if args.layers is None:
            source = projector(tower(padded_pixels, padded_positions).last_hidden_state).numpy()
            source_error = error_metrics(source[None, ...], fixed_reference)

    result: dict[str, object] = {
        "artifact": str(args.artifact),
        "requested_image_size": [args.width, args.height],
        "processor_patch_shape": list(padded_pixels.shape),
        "fixed_patch_shape": list(pixels.shape),
        "fixed_patch_grid": [
            int(positions[0, :, 0].max().item()) + 1,
            int(positions[0, :, 1].max().item()) + 1,
        ],
        "layers": len(wrapper.layers),
        "reference_output_shape": list(fixed_reference.shape),
        "fixed_wrapper_vs_huggingface": source_error,
    }
    if not args.skip_convert:
        result["conversion"] = convert_models(
            wrapper, pixels, args.output, not args.skip_palettize
        )
    if not args.skip_benchmark:
        benchmarks = {}
        paths = {
            "fp16": args.output / "gemma4_e4b_vision_fp16.mlpackage",
            "int4": args.output / "gemma4_e4b_vision_int4.mlpackage",
        }
        for name, path in paths.items():
            if path.exists():
                benchmarks[name] = benchmark_coreml(
                    path, pixels, fixed_reference, args.warmup, args.rounds
                )
        result["benchmarks_cpu_and_ne"] = benchmarks

    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
