# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:35:20 2025

@author: Eleonora Cristina Amico
"""

from setuptools import setup, find_packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
setup(
    name="calibration_ebt3",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={  
        'dev': [        # Dependencies for testing
            'pytest==7.4.0',
            'hypothesis==6.124.0',
            'pytest-cov==6.0.0',
        ],
        'docs': [      # Dependencies for documentation (e.g., Sphinx themes, etc.)
            'Sphinx==5.0.2',
        ],
    },
    author="Eleonora Cristina Amico",
    author_email="eleonora.c.amico@gmail.com",
    description="A package for EBT3 calibration.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EleonoraAmico/Calibration_EBT3",  
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
