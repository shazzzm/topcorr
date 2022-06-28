import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name='topcorr',  
    version='0.14',
    author="Tristan Millington",
    author_email="tristan.millington@gmail.com",
    description="A package for consutructing filtered correlation networks",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/shazzzm/topcorr",
    packages=setuptools.find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],

 )