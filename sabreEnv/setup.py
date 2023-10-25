from setuptools import setup

setup(
    name="sabreEnv",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1"],
    packages=['gymSabre', 'sabre', 'utils'],
)