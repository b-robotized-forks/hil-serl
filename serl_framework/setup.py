"""
Setup configuration for the serl_framework package.
"""
from setuptools import setup


package_name = "serl_framework"

setup(
    name=package_name,
    version="0.0.0",
    packages=[
        package_name,
        f"{package_name}.envs",
        f"{package_name}.utils",
        f"{package_name}.testing",
        f"{package_name}.train",
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Jennifer Buehler",
    maintainer_email="jennifer.e.buehler@gmail.com",
    description="ROS2-free robot-agnostic environments for HIL-SERL.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
