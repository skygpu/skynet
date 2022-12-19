from setuptools import setup, find_packages

setup(
    name='skynet',
    version='0.1.0a6',
    description='Decentralized compute platform',
    author='Guillermo Rodriguez',
    author_email='guillermo@telos.net',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'skynet = skynet.cli:skynet',
            'txt2img = skynet.cli:txt2img',
            'upscale = skynet.cli:upscale'
        ]
    },
    install_requires=['click']
)
