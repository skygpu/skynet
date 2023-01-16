from setuptools import setup, find_packages

from skynet.constants import VERSION

setup(
    name='skynet',
    version=VERSION,
    description='Decentralized compute platform',
    author='Guillermo Rodriguez',
    author_email='guillermo@telos.net',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'skynet = skynet.cli:skynet',
            'txt2img = skynet.cli:txt2img',
            'img2img = skynet.cli:img2img',
            'upscale = skynet.cli:upscale'
        ]
    },
    install_requires=['click']
)
