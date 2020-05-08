import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amino-acid-vae",  # Replace with your own username
    version="0.0.1",
    author="Anonymous",
    author_email="anonymous@relationrx.com",
    description="Amino acid vae",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
