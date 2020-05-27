# -*- coding: utf-8 -*-
"""
package setup

@author: C Heiser
"""
import sys
import os
import io
import setuptools
from setuptools import setup


if __name__ == "__main__":
    import versioneer

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="cNMF",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description="Consensus Non-Negative Matrix Factorization for scRNA-seq data",
        long_description=long_description,
        author="Cody Heiser",
        author_email="codyheiser49@gmail.com",
        url="https://github.com/codyheiser/cNMF",
        install_requires=read("requirements.txt").splitlines(),
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.6",
        entry_points={
          "console_scripts": [
              "cnmf = cNMF.cnmf:main",
              "cnmf_p = cNMF.cnmf_parallel:main",
          ]
      },
    )
