from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension(
    name="MazeGenerator_cython",
    sources=["MazeGenerator_cython.pyx"],
    include_dirs=[np.get_include()],
),
Extension(
    name="MazeWorld_cython",
    sources=["MazeWorld_cython.pyx"],
    include_dirs=[np.get_include()],
)
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
