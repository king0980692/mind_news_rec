import setuptools
from distutils.core import setup
from Cython.Build import cythonize
with open("README.md", "r", encoding="utf-8") as fhand:
    long_description = fhand.read()


setuptools.setup(
    name="pysmore",
    version="0.0.1-dev",
    author="Leon, Chang",
    author_email="king0980692@gmail.com",
    description=("An pytorch version of smore"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cnclabs/pysmore",
    project_urls={
        "Bug Tracker": "https://github.com/cnclabs/pysmore/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules = cythonize('./pysmore/utils/c_alias_method.pyx'),

    install_requires=["tqdm", "torch", "numpy" ,"pandas","tensorboard","tensorboardx","scipy"],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    # scripts=['bin/train.py']

    entry_points={
        "console_scripts": [
            "pysmore_train=pysmore.train:entry_points",
            # "pysmore_rec=pysmore.pred:entry_points",
        ]
    }
)
