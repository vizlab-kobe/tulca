from setuptools import setup

setup(
    name="tulca",
    version="0.1.0",
    packages=["tulca"],
    package_dir={"tulca": "tulca"},
    install_requires=[
        "autograd",
        "numpy",
        "scipy",
        "tensorly",
        "pymanopt",
    ],
    py_modules=["tulca"],
)
