import sys
from distutils.core import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name='PyCS3',
    version='3.0.1',
    author='Martin Millon',
    author_email='martin.millon@epfl.ch',
    description='Python Curve Shifting for python 3',
    packages=["pycs3", "pycs3.gen", "pycs3.spl", "pycs3.regdiff", "pycs3.sim", "pycs3.tdcomb","pycs3.pipe"],
    requires=['numpy', 'matplotlib', 'scipy','sklearn', 'multiprocess'],
    cmdclass={'test': PyTest}
)
