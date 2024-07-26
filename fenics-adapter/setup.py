from setuptools import setup, find_packages


setup(
    name="fenicsxprecice",
    version="0.0.1",
    packages=find_packages(),
    description="FEniCSx-preCICE adapter is a preCICE adapter for the open source computing platform FEniCSx.",
    url="https://github.com/efirvida/fenicsx-adapter",
    author="Eduardo M Firvida Donestevez",
    author_email="efirvida@gmail.org",
    license="LGPL-3.0",
    install_requires=["pyprecice>=3.0.0"],
)
