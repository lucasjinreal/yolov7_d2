import os
import subprocess
import time
from setuptools import find_packages, setup
import io
from os import path


this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "readme.md"), encoding="utf-8") as f:
    long_description = f.read()


version_file = "yolov7/version.py"


def get_version():
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


if __name__ == "__main__":
    setup(
        name="yolov7_d2",
        version=get_version(),
        description="YOLOv7D2 is a high-level training framework based on detectron2",
        long_description="",
        author="LucasJin",
        author_email="jinfagang19@163.com",
        keywords="computer vision, object detection",
        url="https://github.com/jinfagang/yolov7_d2",
        packages=find_packages(exclude=("configs", "tools", "demo", "images")),
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        license="Apache License 2.0",
        zip_safe=False,
    )
