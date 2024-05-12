"""
This file is required for create and distribute as a Python package
"""

from setuptools import setup, find_packages

setup(
    name="opplx",
    version="0.12",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "opplx = ollama.__init__:main",
        ],
    },
    install_requires=[
        "langchain",
        "googlesearch-python",
        "twine",
        "setuptools",
        "wheel",
        "faiss-cpu",
    ],
)
