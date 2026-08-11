import hashlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.base import ensure_fused_sdpa
from mlx_vlm.models.gemma4.vision import apply_multidimensional_rope


QKV_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_qkv_epilogue",
    input_names=[
        "qkv",
        "output_mins",
        "output_maxs",
        "norm_weights",
        "cosine",
        "sine",
    ],
    output_names=["out"],
    source="""
        uint d = thread_position_in_threadgroup.x;
        uint token = threadgroup_position_in_grid.y;
        uint plane = threadgroup_position_in_grid.z;
        uint kind = plane / Heads;
        uint head = plane % Heads;
        uint lane = thread_index_in_simdgroup;
        uint simdgroup = simdgroup_index_in_threadgroup;

        uint hidden = Heads * HeadDim;
        uint input_index = token * (3 * hidden) + kind * hidden + head * HeadDim + d;
        float value = static_cast<float>(qkv[input_index]);
        value = clamp(
            value,
            static_cast<float>(output_mins[kind]),
            static_cast<float>(output_maxs[kind])
        );

        float partial = simd_sum(value * value);
        threadgroup float sums[2];
        if (lane == 0) sums[simdgroup] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            float total = lane < 2 ? sums[lane] : 0.0f;
            total = simd_sum(total);
            if (lane == 0) sums[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float inverse_rms = rsqrt(sums[0] / static_cast<float>(HeadDim) + 1.0e-6f);
        T normalized = static_cast<T>(
            value * inverse_rms
            * static_cast<float>(norm_weights[kind * HeadDim + d])
        );
        threadgroup T normalized_values[HeadDim];
        normalized_values[d] = normalized;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        T result = normalized;
        if (kind < 2) {
            uint channels_per_dimension = HeadDim / 2;
            uint half_channels = channels_per_dimension / 2;
            uint part_d = d % channels_per_dimension;
            uint partner = part_d < half_channels ? d + half_channels : d - half_channels;
            float rotated = part_d < half_channels
                ? -static_cast<float>(normalized_values[partner])
                : static_cast<float>(normalized_values[partner]);
            uint rope_index = token * HeadDim + d;
            T scaled = static_cast<T>(
                static_cast<float>(normalized) * static_cast<float>(cosine[rope_index])
            );
            T rotated_scaled = static_cast<T>(
                rotated * static_cast<float>(sine[rope_index])
            );
            result = static_cast<T>(
                static_cast<float>(scaled) + static_cast<float>(rotated_scaled)
            );
        }
        uint output_index = ((kind * Heads + head) * Length + token) * HeadDim + d;
        out[output_index] = result;
    """,
)


ROPE_LAYOUT_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_rope_layout_epilogue",
    input_names=["q", "k", "v", "cosine", "sine"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source="""
        uint d = thread_position_in_threadgroup.x;
        uint token = threadgroup_position_in_grid.y;
        uint plane = threadgroup_position_in_grid.z;
        uint batch = plane / (3 * Heads);
        uint local_plane = plane % (3 * Heads);
        uint kind = local_plane / Heads;
        uint head = local_plane % Heads;
        uint input_index = (
            (batch * threads_per_grid.y + token) * Heads + head
        ) * HeadDim + d;
        T normalized = kind == 0 ? q[input_index]
            : (kind == 1 ? k[input_index] : v[input_index]);
        T result = normalized;
        if (kind < 2) {
            uint channels = HeadDim / 2;
            uint half_channels = channels / 2;
            uint part_d = d % channels;
            uint partner_d = part_d < half_channels
                ? d + half_channels
                : d - half_channels;
            uint partner_index = (
                (batch * threads_per_grid.y + token) * Heads + head
            ) * HeadDim + partner_d;
            T partner = kind == 0 ? q[partner_index] : k[partner_index];
            float rotated = part_d < half_channels
                ? -static_cast<float>(partner)
                : static_cast<float>(partner);
            uint rope_index = token * HeadDim + d;
            T scaled = static_cast<T>(
                static_cast<float>(normalized) * static_cast<float>(cosine[rope_index])
            );
            T rotated_scaled = static_cast<T>(
                rotated * static_cast<float>(sine[rope_index])
            );
            result = static_cast<T>(
                static_cast<float>(scaled) + static_cast<float>(rotated_scaled)
            );
        }
        uint output_index = (
            (((batch * 3 + kind) * Heads + head) * threads_per_grid.y + token)
            * HeadDim + d
        );
        out[output_index] = result;
    """,
)


SDPA_OUTPUT_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_sdpa_output_epilogue",
    input_names=["input", "minimum", "maximum"],
    output_names=["out"],
    source="""
        uint channel = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        uint head = channel / HeadDim;
        uint d = channel % HeadDim;
        uint input_index = (head * threads_per_grid.y + token) * HeadDim + d;
        float value = static_cast<float>(input[input_index]);
        out[token * Hidden + channel] = static_cast<T>(clamp(
            value,
            static_cast<float>(minimum),
            static_cast<float>(maximum)
        ));
    """,
)


NORM_ROPE_LAYOUT_EPILOGUE = mx.fast.metal_kernel(
    name="gemma4_vision_norm_rope_layout_epilogue",
    input_names=[
        "q", "k", "v",
        "q_inverse_rms", "k_inverse_rms", "v_inverse_rms",
        "q_weight", "k_weight", "cosine", "sine",
    ],
    output_names=["out"],
    source="""
        uint d = thread_position_in_threadgroup.x;
        uint token = threadgroup_position_in_grid.y;
        uint plane = threadgroup_position_in_grid.z;
        uint kind = plane / Heads;
        uint head = plane % Heads;
        uint input_index = (token * Heads + head) * HeadDim + d;
        uint norm_index = token * Heads + head;
        float value = kind == 0 ? static_cast<float>(q[input_index])
            : (kind == 1 ? static_cast<float>(k[input_index])
                         : static_cast<float>(v[input_index]));
        float inverse_rms = kind == 0 ? q_inverse_rms[norm_index]
            : (kind == 1 ? k_inverse_rms[norm_index]
                         : v_inverse_rms[norm_index]);
        float weight = kind == 0 ? static_cast<float>(q_weight[d])
            : (kind == 1 ? static_cast<float>(k_weight[d]) : 1.0f);
        T normalized = static_cast<T>(value * inverse_rms * weight);
        T result = normalized;
        if (kind < 2) {
            uint channels = HeadDim / 2;
            uint half_channels = channels / 2;
            uint part_d = d % channels;
            uint partner_d = part_d < half_channels
                ? d + half_channels
                : d - half_channels;
            uint partner_index = (token * Heads + head) * HeadDim + partner_d;
            float partner_value = kind == 0
                ? static_cast<float>(q[partner_index])
                : static_cast<float>(k[partner_index]);
            float partner_weight = kind == 0
                ? static_cast<float>(q_weight[partner_d])
                : static_cast<float>(k_weight[partner_d]);
            T partner = static_cast<T>(partner_value * inverse_rms * partner_weight);
            float rotated = part_d < half_channels
                ? -static_cast<float>(partner)
                : static_cast<float>(partner);
            uint rope_index = token * HeadDim + d;
            result = static_cast<T>(
                static_cast<float>(normalized) * static_cast<float>(cosine[rope_index])
                + rotated * static_cast<float>(sine[rope_index])
            );
        }
        uint output_index = (plane * threads_per_grid.y + token) * HeadDim + d;
        out[output_index] = result;
    """,
)


_ROPE_CONSTANT_CACHE = {}


def _rope_constants(attention, positions):
    positions_np = np.array(positions)
    key = (
        positions_np.shape,
        hashlib.sha256(positions_np.tobytes()).digest(),
        attention.head_dim,
        attention.rope_base_frequency,
        str(attention.q_norm.weight.dtype),
    )
    if key not in _ROPE_CONSTANT_CACHE:
        dimensions = positions.shape[-1]
        channels = attention.head_dim // dimensions
        half = channels // 2
        exponents = (2.0 / channels) * mx.arange(half).astype(mx.float32)
        timescale = mx.power(attention.rope_base_frequency, exponents)
        cosine_parts = []
        sine_parts = []
        for dimension in range(dimensions):
            sinusoid = (
                positions[..., dimension : dimension + 1].astype(mx.float32)
                / timescale
            )
            cosine = mx.cos(sinusoid)
            sine = mx.sin(sinusoid)
            cosine_parts.append(
                mx.concatenate([cosine, cosine], axis=-1).astype(
                    attention.q_norm.weight.dtype
                )
            )
            sine_parts.append(
                mx.concatenate([sine, sine], axis=-1).astype(
                    attention.q_norm.weight.dtype
                )
            )
        cosine = mx.concatenate(cosine_parts, axis=-1).reshape(
            positions.shape[1], attention.head_dim
        )
        sine = mx.concatenate(sine_parts, axis=-1).reshape(
            positions.shape[1], attention.head_dim
        )
        mx.eval(cosine, sine)
        _ROPE_CONSTANT_CACHE[key] = (cosine, sine)
    return _ROPE_CONSTANT_CACHE[key]


class EpilogueFusedVisionAttention(nn.Module):
    def __init__(self, attention, *, fuse_norm: bool, fuse_output: bool = False):
        super().__init__()
        if attention.num_heads != attention.num_kv_heads:
            raise ValueError("Epilogue fusion requires equal Q and KV head counts")
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.rope_base_frequency = attention.rope_base_frequency
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.v_norm = attention._v_norm
        self.o_proj = attention.o_proj
        self.fuse_norm = fuse_norm
        self.fuse_output = fuse_output
        self._optimized_rope_cache = {}
        self._active_rope_constants = None

    def __call__(self, value, positions, mask=None):
        batch, length, _ = value.shape
        if batch != 1 and (self.fuse_norm or self.fuse_output):
            raise ValueError(
                "Batched epilogue fusion supports RoPE/layout fusion only"
            )
        q = self.q_proj(value).reshape(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(value).reshape(batch, length, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch, length, self.num_heads, self.head_dim)
        if isinstance(positions, tuple):
            cosine, sine = positions
        elif self._active_rope_constants is None:
            cosine, sine = _rope_constants(self, positions)
        else:
            cosine, sine = self._active_rope_constants
        template = [
            ("T", q.dtype),
            ("Heads", self.num_heads),
            ("HeadDim", self.head_dim),
        ]
        if self.fuse_norm:
            q = mx.contiguous(q)
            k = mx.contiguous(k)
            v = mx.contiguous(v)

            def inverse_rms(array):
                array_float = array.astype(mx.float32)
                variance = mx.mean(array_float**2, axis=-1, keepdims=True)
                return mx.rsqrt(variance + self.q_norm.eps)

            processed = NORM_ROPE_LAYOUT_EPILOGUE(
                inputs=[
                    q, k, v,
                    inverse_rms(q), inverse_rms(k), inverse_rms(v),
                    self.q_norm.weight, self.k_norm.weight,
                    cosine, sine,
                ],
                output_shapes=[(3, self.num_heads, length, self.head_dim)],
                output_dtypes=[q.dtype],
                grid=(self.head_dim, length, 3 * self.num_heads),
                threadgroup=(self.head_dim, 1, 1),
                template=template,
            )[0]
        else:
            processed = ROPE_LAYOUT_EPILOGUE(
                inputs=[
                    self.q_norm(q), self.k_norm(k), self.v_norm(v),
                    cosine, sine,
                ],
                output_shapes=[
                    (batch, 3, self.num_heads, length, self.head_dim)
                ],
                output_dtypes=[q.dtype],
                grid=(self.head_dim, length, batch * 3 * self.num_heads),
                threadgroup=(self.head_dim, 1, 1),
                template=template,
            )[0]
        output = ensure_fused_sdpa(
            processed[:, 0], processed[:, 1], processed[:, 2],
            scale=1.0,
            mask=mask,
        )
        if self.fuse_output:
            output = SDPA_OUTPUT_EPILOGUE(
                inputs=[output, self.o_proj.input_min, self.o_proj.input_max],
                output_shapes=[(batch, length, self.num_heads * self.head_dim)],
                output_dtypes=[output.dtype],
                grid=(self.num_heads * self.head_dim, length, 1),
                threadgroup=(256, 1, 1),
                template=[
                    ("T", output.dtype),
                    ("HeadDim", self.head_dim),
                    ("Hidden", self.num_heads * self.head_dim),
                ],
            )[0]
            output = self.o_proj.linear(output)
            return mx.clip(
                output,
                self.o_proj.output_min,
                self.o_proj.output_max,
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


def _shapeless_multidimensional_rope(inputs, positions, base_frequency):
    head_dim = inputs.shape[-1]
    if isinstance(positions, tuple):
        cosine, sine = positions
        dimensions = cosine.shape[-2]
        channels = cosine.shape[-1]
        half_channels = channels // 2
    elif positions.shape[-1] == 2:
        dimensions = positions.shape[-1]
        channels = head_dim // dimensions
        half_channels = channels // 2
        exponents = (2.0 / channels) * mx.arange(half_channels).astype(mx.float32)
        timescale = mx.power(base_frequency, exponents)
        sinusoid = positions.astype(mx.float32)[..., None] / timescale
        cosine = mx.concatenate([mx.cos(sinusoid), mx.cos(sinusoid)], axis=-1)
        sine = mx.concatenate([mx.sin(sinusoid), mx.sin(sinusoid)], axis=-1)
    head_axis = 1 if inputs.ndim == 3 else 2
    cosine = mx.expand_dims(cosine.astype(inputs.dtype), axis=head_axis)
    sine = mx.expand_dims(sine.astype(inputs.dtype), axis=head_axis)
    shaped = mx.unflatten(inputs, -1, (dimensions, channels))
    signs = mx.array(
        [-1] * half_channels + [1] * half_channels,
        dtype=inputs.dtype,
    )
    rotated = mx.roll(shaped, half_channels, axis=-1) * signs
    return mx.flatten(
        shaped * cosine + rotated * sine,
        start_axis=-2,
        end_axis=-1,
    )


class ShapelessVisionAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.rope_base_frequency = attention.rope_base_frequency
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.v_norm = attention._v_norm
        self.o_proj = attention.o_proj

    def __call__(self, value, positions, mask=None):
        q = mx.unflatten(
            self.q_proj(value), -1, (self.num_heads, self.head_dim)
        )
        k = mx.unflatten(
            self.k_proj(value), -1, (self.num_kv_heads, self.head_dim)
        )
        v = mx.unflatten(
            self.v_proj(value), -1, (self.num_kv_heads, self.head_dim)
        )
        q = _shapeless_multidimensional_rope(
            self.q_norm(q), positions, self.rope_base_frequency
        )
        k = _shapeless_multidimensional_rope(
            self.k_norm(k), positions, self.rope_base_frequency
        )
        v = self.v_norm(v)
        if value.ndim == 2:
            output = mx.fast.scaled_dot_product_attention(
                mx.expand_dims(q.transpose(1, 0, 2), 0),
                mx.expand_dims(k.transpose(1, 0, 2), 0),
                mx.expand_dims(v.transpose(1, 0, 2), 0),
                scale=1.0,
                mask=mask,
            )
            output = mx.flatten(
                mx.squeeze(output, axis=0).transpose(1, 0, 2),
                start_axis=-2,
                end_axis=-1,
            )
        else:
            output = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                scale=1.0,
                mask=mask,
            )
            output = mx.flatten(
                output.transpose(0, 2, 1, 3),
                start_axis=-2,
                end_axis=-1,
            )
        return self.o_proj(output)


class FusedVisionAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        for projection in (attention.k_proj, attention.v_proj):
            if not mx.array_equal(
                attention.q_proj.input_min, projection.input_min
            ).item() or not mx.array_equal(
                attention.q_proj.input_max, projection.input_max
            ).item():
                raise ValueError("Q/K/V input clipping bounds must match")
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.rope_base_frequency = attention.rope_base_frequency
        self.input_min = attention.q_proj.input_min
        self.input_max = attention.q_proj.input_max
        self.q_output_min = attention.q_proj.output_min
        self.q_output_max = attention.q_proj.output_max
        self.k_output_min = attention.k_proj.output_min
        self.k_output_max = attention.k_proj.output_max
        self.v_output_min = attention.v_proj.output_min
        self.v_output_max = attention.v_proj.output_max
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.v_norm = attention._v_norm
        self.o_proj = attention.o_proj
        self.qkv_weight = mx.concatenate(
            [
                attention.q_proj.linear.weight,
                attention.k_proj.linear.weight,
                attention.v_proj.linear.weight,
            ],
            axis=0,
        )

    def __call__(self, value, positions, mask=None):
        batch, length, _ = value.shape
        value = mx.clip(value, self.input_min, self.input_max)
        q, k, v = mx.split(value @ self.qkv_weight.T, 3, axis=-1)
        q = mx.clip(q, self.q_output_min, self.q_output_max)
        k = mx.clip(k, self.k_output_min, self.k_output_max)
        v = mx.clip(v, self.v_output_min, self.v_output_max)
        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_kv_heads, self.head_dim)
        q = apply_multidimensional_rope(
            self.q_norm(q),
            positions,
            self.rope_base_frequency,
        )
        k = apply_multidimensional_rope(
            self.k_norm(k),
            positions,
            self.rope_base_frequency,
        )
        v = self.v_norm(v)
        output = ensure_fused_sdpa(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            scale=1.0,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class ReassociatedFusedVisionAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        if attention.num_heads != attention.num_kv_heads or attention.head_dim != 64:
            raise ValueError("QKV epilogue requires 64D full multi-head attention")
        for projection in (attention.k_proj, attention.v_proj):
            if not mx.array_equal(
                attention.q_proj.input_min, projection.input_min
            ).item() or not mx.array_equal(
                attention.q_proj.input_max, projection.input_max
            ).item():
                raise ValueError("Q/K/V input clipping bounds must match")
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.rope_base_frequency = attention.rope_base_frequency
        self.input_min = attention.q_proj.input_min
        self.input_max = attention.q_proj.input_max
        self.o_proj = attention.o_proj
        self.qkv_weight = mx.concatenate(
            [
                attention.q_proj.linear.weight,
                attention.k_proj.linear.weight,
                attention.v_proj.linear.weight,
            ],
            axis=0,
        )
        self.output_mins = mx.stack(
            [
                attention.q_proj.output_min,
                attention.k_proj.output_min,
                attention.v_proj.output_min,
            ]
        )
        self.output_maxs = mx.stack(
            [
                attention.q_proj.output_max,
                attention.k_proj.output_max,
                attention.v_proj.output_max,
            ]
        )
        self.norm_weights = mx.stack(
            [
                attention.q_norm.weight,
                attention.k_norm.weight,
                mx.ones(
                    (self.head_dim,),
                    dtype=attention.q_norm.weight.dtype,
                ),
            ]
        )
        self._rope_cache = {}
        mx.eval(
            self.qkv_weight,
            self.output_mins,
            self.output_maxs,
            self.norm_weights,
        )

    def _rope(self, positions):
        if isinstance(positions, tuple):
            return positions
        key = (
            *positions.shape,
            int(mx.max(positions[..., 0]).item()),
            int(mx.max(positions[..., 1]).item()),
            int(mx.min(positions).item()),
        )
        if key not in self._rope_cache:
            dimensions = positions.shape[-1]
            channels = self.head_dim // dimensions
            half = channels // 2
            exponents = (2.0 / channels) * mx.arange(half).astype(mx.float32)
            timescale = mx.power(self.rope_base_frequency, exponents)
            cosine_parts = []
            sine_parts = []
            for dimension in range(dimensions):
                sinusoid = (
                    positions[..., dimension : dimension + 1].astype(mx.float32)
                    / timescale
                )
                cosine = mx.cos(sinusoid)
                sine = mx.sin(sinusoid)
                cosine_parts.append(
                    mx.concatenate([cosine, cosine], axis=-1).astype(
                        self.qkv_weight.dtype
                    )
                )
                sine_parts.append(
                    mx.concatenate([sine, sine], axis=-1).astype(
                        self.qkv_weight.dtype
                    )
                )
            cosine = mx.concatenate(cosine_parts, axis=-1).reshape(
                positions.shape[1], self.head_dim
            )
            sine = mx.concatenate(sine_parts, axis=-1).reshape(
                positions.shape[1], self.head_dim
            )
            mx.eval(cosine, sine)
            self._rope_cache[key] = (cosine, sine)
        return self._rope_cache[key]

    def __call__(self, value, positions, mask=None):
        batch, length, _ = value.shape
        if batch != 1:
            raise ValueError("QKV epilogue requires batch size 1")
        value = mx.clip(value, self.input_min, self.input_max)
        qkv = value @ self.qkv_weight.T
        cosine, sine = self._rope(positions)
        processed = QKV_EPILOGUE(
            inputs=[
                qkv,
                self.output_mins,
                self.output_maxs,
                self.norm_weights,
                cosine,
                sine,
            ],
            output_shapes=[(3, self.num_heads, length, self.head_dim)],
            output_dtypes=[qkv.dtype],
            grid=(self.head_dim, length, 3 * self.num_heads),
            threadgroup=(self.head_dim, 1, 1),
            template=[
                ("T", qkv.dtype),
                ("Heads", self.num_heads),
                ("HeadDim", self.head_dim),
                ("Length", length),
            ],
        )[0]
        output = ensure_fused_sdpa(
            processed[0][None],
            processed[1][None],
            processed[2][None],
            scale=1.0,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


def optimize_gemma4_positions(tower) -> None:
    table = tower.patch_embedder.position_embedding_table
    position_size = tower.patch_embedder.position_embedding_size

    def gathered_position_embeddings(patch_positions, padding_positions):
        positions = mx.clip(patch_positions, 0, position_size - 1)
        embeddings = table[0, positions[..., 0]] + table[1, positions[..., 1]]
        return mx.where(
            mx.expand_dims(padding_positions, -1),
            0.0,
            embeddings,
        )

    tower.patch_embedder._position_embeddings = gathered_position_embeddings


def optimize_gemma4_shapeless_positions(tower) -> None:
    table = tower.patch_embedder.position_embedding_table
    position_size = tower.patch_embedder.position_embedding_size
    x_table = mx.take(table, mx.array(0), axis=0)
    y_table = mx.take(table, mx.array(1), axis=0)

    def gathered_position_embeddings(patch_positions, padding_positions):
        positions = mx.clip(patch_positions, 0, position_size - 1)
        x_positions = mx.take(positions, mx.array(0), axis=-1)
        y_positions = mx.take(positions, mx.array(1), axis=-1)
        embeddings = x_table[x_positions] + y_table[y_positions]
        return mx.where(
            mx.expand_dims(padding_positions, -1),
            0.0,
            embeddings,
        )

    tower.patch_embedder._position_embeddings = gathered_position_embeddings


def encode_gemma4_unpadded_batch1(tower, projector, pixels):
    if pixels.shape[0] != 1:
        raise ValueError("Unpadded Gemma 4 vision encoding requires batch size 1")
    _, _, height, width = pixels.shape
    pool_squared = tower.pooling_kernel_size**2
    patch_count = (height // tower.patch_size) * (width // tower.patch_size)
    output_length = patch_count // pool_squared
    positions, padding, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=patch_count,
    )
    positions = mx.array(np.expand_dims(positions, 0))
    padding = mx.array(np.expand_dims(padding, 0))
    hidden = tower.patch_embedder(pixels, positions, padding)
    hidden = tower.encoder(hidden, positions, mask=None)
    hidden, _ = tower.pooler(
        hidden,
        positions,
        padding,
        output_length=output_length,
    )
    if tower.config.standardize:
        hidden = (hidden - tower.std_bias) * tower.std_scale
    return projector(hidden)


def make_segmented_gemma4_encoder(
    tower,
    projector,
    segment_size,
    evaluate_segments=False,
):
    patch = mx.compile(
        lambda pixels, positions, padding: tower.patch_embedder(
            pixels,
            positions,
            padding,
        )
    )
    segments = []
    fused_rope = any(
        isinstance(
            layer.self_attn,
            (EpilogueFusedVisionAttention, ReassociatedFusedVisionAttention),
        )
        for layer in tower.encoder.layers
    )
    for start in range(0, len(tower.encoder.layers), segment_size):
        layers = tuple(tower.encoder.layers[start : start + segment_size])

        if fused_rope:
            def run_segment(hidden, cosine, sine, current_layers=layers):
                for layer in current_layers:
                    hidden = layer(hidden, (cosine, sine), None)
                return hidden
        else:
            def run_segment(hidden, positions, current_layers=layers):
                for layer in current_layers:
                    hidden = layer(hidden, positions, None)
                return hidden

        segments.append(mx.compile(run_segment))

    def finish(hidden, positions, padding):
        output_length = hidden.shape[1] // (tower.pooling_kernel_size**2)
        hidden, _ = tower.pooler(
            hidden,
            positions,
            padding,
            output_length=output_length,
        )
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        return hidden if projector is None else projector(hidden)

    finish = mx.compile(finish)
    def encode(pixels):
        batch, _, height, width = pixels.shape
        patch_count = (height // tower.patch_size) * (width // tower.patch_size)
        positions, padding, _ = tower._patch_positions_single(
            height,
            width,
            max_patches=patch_count,
        )
        positions = mx.array(np.expand_dims(positions, 0))
        padding = mx.array(np.expand_dims(padding, 0))
        rope_constants = prepare_gemma4_rope_constants(tower, positions)
        if batch > 1:
            positions = mx.broadcast_to(
                positions,
                (batch, positions.shape[1], positions.shape[2]),
            )
            padding = mx.broadcast_to(padding, (batch, padding.shape[1]))
        hidden = patch(pixels, positions, padding)
        for segment in segments:
            if fused_rope:
                hidden = segment(hidden, *rope_constants)
            else:
                hidden = segment(hidden, positions)
            if evaluate_segments:
                mx.eval(hidden)
                mx.synchronize()
        return finish(hidden, positions, padding)

    return encode


def prepare_gemma4_rope_constants(tower, positions):
    pair = None
    for layer in tower.encoder.layers:
        attention = layer.self_attn
        if isinstance(attention, EpilogueFusedVisionAttention):
            if pair is None:
                pair = _rope_constants(attention, positions)
            attention._active_rope_constants = pair
        elif isinstance(attention, ReassociatedFusedVisionAttention):
            if pair is None:
                pair = attention._rope(positions)
    return pair


def exact_pool_gemma4_unpadded(tower, hidden, patch_height, patch_width):
    pool = tower.pooling_kernel_size
    output_width = patch_width // pool
    output_length = (patch_height // pool) * output_width
    y, x = np.indices((patch_height, patch_width))
    kernel_ids = (x // pool) + output_width * (y // pool)
    weights_np = np.eye(output_length, dtype=np.float32)[kernel_ids.reshape(-1)]
    weights = mx.array(weights_np[None]) / (pool**2)
    pooled = mx.einsum("bLl,bLd->bld", weights, hidden).astype(hidden.dtype)
    return pooled * tower.pooler.root_hidden_size


def encode_gemma4_exact_pool_batch1(tower, projector, pixels):
    if pixels.shape[0] != 1:
        raise ValueError("Exact-pooled Gemma 4 vision encoding requires batch size 1")
    _, _, height, width = pixels.shape
    patch_height = height // tower.patch_size
    patch_width = width // tower.patch_size
    patch_count = patch_height * patch_width
    positions, padding, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=patch_count,
    )
    positions = mx.array(np.expand_dims(positions, 0))
    padding = mx.array(np.expand_dims(padding, 0))
    hidden = tower.patch_embedder(pixels, positions, padding)
    hidden = tower.encoder(hidden, positions, mask=None)
    hidden = exact_pool_gemma4_unpadded(
        tower,
        hidden,
        patch_height,
        patch_width,
    )
    if tower.config.standardize:
        hidden = (hidden - tower.std_bias) * tower.std_scale
    return projector(hidden)


def gemma4_unpadded_inputs(tower, height, width):
    patch_height = height // tower.patch_size
    patch_width = width // tower.patch_size
    patch_count = patch_height * patch_width
    positions, padding, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=patch_count,
    )
    output_width = patch_width // tower.pooling_kernel_size
    output_length = (patch_height // tower.pooling_kernel_size) * output_width
    y, x = np.indices((patch_height, patch_width))
    kernel_ids = (x // tower.pooling_kernel_size) + output_width * (
        y // tower.pooling_kernel_size
    )
    weights = np.eye(output_length, dtype=np.float32)[kernel_ids.reshape(-1)]
    return (
        mx.array(positions[None]),
        mx.array(padding[None]),
        mx.array(weights[None]) / (tower.pooling_kernel_size**2),
    )


def gemma4_rope_constants(tower, positions):
    if positions.ndim == 3:
        positions = mx.squeeze(positions, axis=0)
    channels = tower.config.head_dim // positions.shape[-1]
    half_channels = channels // 2
    exponents = (2.0 / channels) * mx.arange(half_channels).astype(mx.float32)
    timescale = mx.power(
        tower.config.rope_parameters["rope_theta"],
        exponents,
    )
    sinusoid = positions.astype(mx.float32)[..., None] / timescale
    cosine = mx.concatenate([mx.cos(sinusoid), mx.cos(sinusoid)], axis=-1)
    sine = mx.concatenate([mx.sin(sinusoid), mx.sin(sinusoid)], axis=-1)
    dtype = tower.patch_embedder.position_embedding_table.dtype
    cosine = cosine.astype(dtype)
    sine = sine.astype(dtype)
    mx.eval(cosine, sine)
    return cosine, sine


def encode_gemma4_shapeless_batch1(
    tower,
    projector,
    pixels,
    positions,
    padding,
    pool_weights,
):
    hidden = tower.patch_embedder(pixels, positions, padding)
    hidden = tower.encoder(hidden, positions, mask=None)
    hidden = mx.einsum("bLl,bLd->bld", pool_weights, hidden).astype(hidden.dtype)
    hidden = hidden * tower.pooler.root_hidden_size
    if tower.config.standardize:
        hidden = (hidden - tower.std_bias) * tower.std_scale
    return projector(hidden)


def encode_gemma4_shapeless_hidden(
    tower,
    projector,
    hidden,
    cosine,
    sine,
    pool_weights,
):
    hidden = tower.encoder(hidden, (cosine, sine), mask=None)
    hidden = mx.einsum("Ll,Ld->ld", pool_weights, hidden).astype(hidden.dtype)
    hidden = hidden * tower.pooler.root_hidden_size
    if tower.config.standardize:
        hidden = (hidden - tower.std_bias) * tower.std_scale
    return mx.expand_dims(projector(hidden), 0)


def encode_gemma4_reshape_pool_batch1(tower, projector, pixels):
    if pixels.shape[0] != 1:
        raise ValueError("Reshape-pooled Gemma 4 vision encoding requires batch size 1")
    _, _, height, width = pixels.shape
    pool = tower.pooling_kernel_size
    patch_height = height // tower.patch_size
    patch_width = width // tower.patch_size
    patch_count = patch_height * patch_width
    positions, padding, _ = tower._patch_positions_single(
        height,
        width,
        max_patches=patch_count,
    )
    positions = mx.array(np.expand_dims(positions, 0))
    padding = mx.array(np.expand_dims(padding, 0))
    hidden = tower.patch_embedder(pixels, positions, padding)
    hidden = tower.encoder(hidden, positions, mask=None)
    hidden_size = hidden.shape[-1]
    hidden = hidden.reshape(
        1,
        patch_height // pool,
        pool,
        patch_width // pool,
        pool,
        hidden_size,
    )
    hidden = hidden.astype(mx.float32).mean(axis=(2, 4))
    hidden = hidden.reshape(1, patch_count // (pool**2), hidden_size).astype(
        tower.patch_embedder.position_embedding_table.dtype
    )
    hidden = hidden * tower.pooler.root_hidden_size
    if tower.config.standardize:
        hidden = (hidden - tower.std_bias) * tower.std_scale
    return projector(hidden)


def fuse_gemma4_qkv(tower) -> None:
    for layer in tower.encoder.layers:
        layer.self_attn = FusedVisionAttention(layer.self_attn)
    mx.eval(*(layer.self_attn.qkv_weight for layer in tower.encoder.layers))


def fuse_gemma4_qkv_epilogue(tower) -> None:
    for layer in tower.encoder.layers:
        layer.self_attn = ReassociatedFusedVisionAttention(layer.self_attn)


def fuse_gemma4_rope_layout(tower, layer_count=None) -> None:
    layers = tower.encoder.layers[:layer_count]
    for layer in layers:
        layer.self_attn = EpilogueFusedVisionAttention(
            layer.self_attn,
            fuse_norm=False,
        )


def fuse_gemma4_post_reduction_epilogue(tower) -> None:
    for layer in tower.encoder.layers:
        layer.self_attn = EpilogueFusedVisionAttention(
            layer.self_attn,
            fuse_norm=True,
        )


def fuse_gemma4_rope_and_output_layout(tower) -> None:
    for layer in tower.encoder.layers:
        layer.self_attn = EpilogueFusedVisionAttention(
            layer.self_attn,
            fuse_norm=False,
            fuse_output=True,
        )


def optimize_gemma4_shapeless_rope(tower) -> None:
    for layer in tower.encoder.layers:
        layer.self_attn = ShapelessVisionAttention(layer.self_attn)


def optimize_gemma4_vision(tower) -> None:
    optimize_gemma4_positions(tower)
    fuse_gemma4_qkv(tower)
