from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements() -> List[str]:
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    requirements = [r for r in requirements if r != HYPHEN_E_DOT]
    return requirements

setup(
    name='diamond_price_prediction',
    version='0.1.0',
    author='Omkar Kulkarni',
    author_email='omkarsk98@gmail.com',
    install_requires=get_requirements(),
    packages=find_packages(),
)