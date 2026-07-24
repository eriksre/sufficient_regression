"""Build the fused native update kernel.

Cython keeps the public estimator state as NumPy arrays while compiling the
strictly sequential rank-one loops to native code. This avoids a second state
model and keeps dense batch initialization on its already-optimized BLAS path.
"""

from setuptools import Extension, setup

from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        [
            Extension(
                "sufficient_regression._native",
                ["src/sufficient_regression/_native.pyx"],
            )
        ],
        compiler_directives={
            "boundscheck": False,
            "cdivision": True,
            "initializedcheck": False,
            "language_level": 3,
            "nonecheck": False,
            "wraparound": False,
        },
        build_dir="build/cython",
    )
)
