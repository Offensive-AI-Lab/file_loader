import setuptools
setuptools.setup(
    name="file_loader",
    version="0.0.29",
    author="MABADATA",
    author_email="mabadatabgu@gmail.com",

    description="Handle files from local and form bucket",
    url="https://github.com/MABADATA/file_loader",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    dependency_links=[
        'https://pypi.python.org/simple'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
