# setup.py
from setuptools import setup, find_packages
setup(
    name="fisheyetools",  # Package name for installation
    version="0.1",
    packages=find_packages(),  # Finds all subpackages within fisheye_tools
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
