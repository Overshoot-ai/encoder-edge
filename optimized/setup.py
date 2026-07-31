from setuptools import setup

setup(
    name="cross-device-gemma-vllm",
    version="0.1.0",
    packages=["optimized_vllm"],
    entry_points={
        "vllm.general_plugins": [
            "cross_device_gemma = optimized_vllm:register",
        ]
    },
)
