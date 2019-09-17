import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ofegplots",
    version="0.0.9",
    author="Y.Ge",
    author_email="y.ge1222@gmail.com",
    description="OpenFOAM postprocessing using pyvista and matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uqyge/ofegplots",
    packages=setuptools.find_packages(),
    install_requires=["pyvista", "matplotlib", "cantera"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

