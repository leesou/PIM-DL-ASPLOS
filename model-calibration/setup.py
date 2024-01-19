from struct import pack
from setuptools import setup, find_packages
from LUTNeuro import __version__
setup(
    name='LUTNeuro',
    packages=find_packages(exclude=['']),
    version=__version__
)
