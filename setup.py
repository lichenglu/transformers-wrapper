import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformers_wrapper",
    version="0.0.1",
    author="Chenglu Li",
    author_email="kabelee92@gmail.com",
    description="A wrapper that helps you get started with pytorch transformers quickly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/transformers_wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
