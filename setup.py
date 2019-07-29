from setuptools import setup, find_packages

setup(
    name="tf_morphology",
    version="0.1",
    description="python library for simple grayscale and binary morphological operations in tensorflow",
    url="https://github.com/theRealSuperMario/tf_morphology",
    author="Sandro Braun",
    author_email="supermario94123@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["tensorflow", "scikit-image"],
    zip_safe=False,
)
