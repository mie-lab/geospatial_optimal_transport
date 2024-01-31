"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="geoemd",
    version="0.0.1",
    description="Optimal Transport for spatiotemporal predictions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MIE Lab",
    author_email=("nwiedemann@ethz.ch"),
    license="MIT",
    url="https://github.com/mie-lab/geospatial_optimal_transport.git",
    install_requires=[
        "numpy",
        "geopandas",
        "matplotlib",
        "torch",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.8",
    scripts=scripts,
)
