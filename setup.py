"""Setup script for Deep Learning for Clinical Neuro-Oncology."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brain-metastasis-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning pipeline for automated detection of brain metastases from CT scans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/brain-metastasis-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "brain-met-train=scripts.train:main",
            "brain-met-inference=scripts.inference:main",
            "brain-met-preprocess=scripts.preprocess_data:main",
        ],
    },
)
