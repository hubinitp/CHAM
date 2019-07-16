from distutils.core import setup
from Cython.Build import cythonize

setup(
        name='jinkeloid',
        ext_modules = cythonize(['ZHfunc_cython.pyx', 'NFWfunc_cython.pyx']),
)
