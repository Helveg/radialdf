import setuptools, sys, os

with open(os.path.join(os.path.dirname(__file__), "radialdf", "__init__.py"), "r") as f:
    for line in f:
        if "__version__ = " in line:
            exec(line.strip())
            break

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="radialdf",
    version=__version__,
    author="Robin De Schepper",
    author_email="robingilbert.deschepper@unipv.it",
    description="A package to calculate the radial distribution function of particles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helveg/radialdf",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "scipy>=1.5.4",
        "numpy>=1.19"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
