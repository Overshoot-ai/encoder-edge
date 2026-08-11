from vllm import ModelRegistry


def register() -> None:
    ModelRegistry.register_model(
        "CrossDeviceGemma4ForConditionalGeneration",
        "optimized_vllm.model:CrossDeviceGemma4ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "CrossDeviceGemma4UnifiedForConditionalGeneration",
        "optimized_vllm.model:CrossDeviceGemma4UnifiedForConditionalGeneration",
    )
