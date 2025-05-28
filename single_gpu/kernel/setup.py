from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="temporal_fusion_kernel",
    include_dirs = ["include"],
    ext_modules = [
        CUDAExtension(
            name = "temporal_fusion_kernel",
            sources = [
                "src/neuron.cpp", 
                "src/lif_kernel.cu", "src/ternary_lif_kernel.cu",
            ],
        )
    ],
    cmdclass = {
        "build_ext": BuildExtension,
    }
)
