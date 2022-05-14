from setuptools import find_packages, setup

__version__ = '0.0.1'
URL = 'https://github.com/marcolatella/NNsimple'

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'numpy',
    'pandas',
    'pytorch_lightning>=1.5',
    'torchvision',
    'torchaudio',
    'torchmetrics',
    'PyYAML',
    'torch>=1.11',
    'tqdm',
]

full_install_requires = [
    'matplotlib'
]

setup(
    name='nnsimple',
    version=__version__,
    description='A PyTorch library for easly implementing neural network models.',
    author='Marco Latella',
    author_email='mrclatella@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license="MIT",
    keywords=[
        'pytorch',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'full': full_install_requires,
    },
    packages=find_packages(exclude=['examples*']),
)
