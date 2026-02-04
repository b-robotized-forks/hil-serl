from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.2",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "zmq",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace@git+https://github.com/JenniferBuehler/agentlace.git@d003c97002fdd317fd92f06c271d28b074c406d9",
    ],
    packages=find_packages(),
    zip_safe=False,
)
