from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpp_mlp',
    ext_modules=[
        CppExtension(
            name='cpp_mlp',
            sources=['cpp_mlp.cpp']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
