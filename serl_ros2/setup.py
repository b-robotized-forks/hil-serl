from glob import glob
from setuptools import setup

package_name = "serl_ros2"
python_package = "serl_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=[python_package],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "serl-framework"],
    zip_safe=True,
    maintainer="Jennifer Buehler",
    maintainer_email="jennifer.e.buehler@gmail.com",
    description="ROS2 adapter package for HIL-SERL.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "serl_ros2_smoke=serl_ros2.smoke:main",
            "joy_mux=serl_ros2.joy_mux:main",
            "keyboard_joy=serl_ros2.keyboard_joy:main",
            "pose_to_joy=serl_ros2.pose_to_joy:main",
        ],
    },
)
