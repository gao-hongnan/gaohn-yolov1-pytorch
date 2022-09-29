# Setup installation for the application

from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

with open(Path(BASE_DIR, "dev_requirements.txt")) as file:
    dev_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="gaohn-yolov1",
    version="0.1",
    license="MIT",
    description="YOLOv1 Implementation in PyTorch.",
    author="Hongnan G",
    author_email="reighns.sjr.sjh.gr.twt.hrj@gmail.com",
    url="",
    keywords=["machine-learning", "deep-learning", "object-detection"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=[required_packages],
    extras_require={"dev": dev_packages},
    dependency_links=[],
    entry_points={
        "console_scripts": [
            "gaohn_yolo= src.main:app",
        ],
    },
)
