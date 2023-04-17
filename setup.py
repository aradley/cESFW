import setuptools
from glob import glob
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setuptools.setup(
    name="cESFW",
    version="0.0.1",
    author="Arthur Radley",
    author_email="arthur_radley@hotmail.co.uk",
    description="cESFW",
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=[script for script in glob('ffaves/*') if os.path.isfile(script)],
    url="https://github.com/aradley/cESFW",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)