from setuptools import setup, find_packages

"""Package setup for transformer_architecture."""
setup(
    name='transformer_architecture',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    author='Manpinder Singh',
    author_email='manpindersingh417@gmail.com',
    description='A clean, modular and well-tested reimplementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) from scratch',
    license='MIT'
)