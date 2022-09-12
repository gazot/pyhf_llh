from os import path
from setuptools import setup, find_packages


setup(
    name="pyhf_llh",
    version="0.1.0",
    packages=find_packages(),
    license="MIT",
    author="Christian Gajek",
    author_email="Christian.Gajek@tum.de",
    description="Wrapper for pyhf log likelihoods with jax compability",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pyhf",
        "jax"
    ],
)
