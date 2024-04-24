from setuptools import setup, find_packages

setup(
    name='opplx',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'opplx = ollama.__init__:main',
        ],
    },
    install_requires=[
        'langchain',
        'googlesearch-python',
        'twine',
        'setuptools',
        'wheel'
    ],
)