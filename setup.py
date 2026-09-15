import os

import numpy
import pufferlib
from setuptools import Extension, setup

setup(
    name='splendor',
    version='0.1.0',
    packages=['splendor'],
    ext_modules=[
        Extension(
            'splendor.binding',
            sources=['splendor/binding.c'],
            depends=['splendor/splendor.h', 'splendor/game.h'],  # rebuild when the header changes
            include_dirs=[
                numpy.get_include(),
                os.path.join(os.path.dirname(pufferlib.__file__), 'ocean'),
            ],
            extra_compile_args=[
                '-O3',
                '-Wno-unused-function',
                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
            ],
        )
    ],
)
