from setuptools import setup

setup(
    name="tulca",
    version="0.1.0",
    packages=["tulca"],
    package_dir={"tulca": "tulca"},
    install_requires=["numpy", "scipy", "tensorly", "factor-analyzer", "pymanopt"],
    py_modules=["tulca"],
)
