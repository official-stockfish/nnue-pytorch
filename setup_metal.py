"""Build the Metal sparse-linear extension ahead of time.

Usage:
    python setup_metal.py install          # install into site-packages
    python setup_metal.py build_ext -i     # build in-place (for development)
"""

import sys

if sys.platform != "darwin":
    sys.exit("Metal extension is only supported on macOS.")

from distutils.unixccompiler import UnixCCompiler

if ".mm" not in UnixCCompiler.src_extensions:
    UnixCCompiler.src_extensions.append(".mm")
    UnixCCompiler.language_map[".mm"] = "objc"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

torch_lib_dir = library_paths()[0]

setup(
    name="sparse_linear_metal_cpp",
    ext_modules=[
        CppExtension(
            name="sparse_linear_metal_cpp",
            sources=[
                "model/modules/feature_transformer/metal/sparse_linear.mm",
            ],
            extra_compile_args=["-std=c++17"],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-Wl,-rpath," + torch_lib_dir,
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
