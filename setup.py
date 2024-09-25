from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="SENTISCOPE",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "nltk",
        "flair",
        "requests"
    ],
    author="Youngwon Cho",
    author_email="youngwon.cho@tum.de",
    description="A package for financial news sentiment analysis and entity extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kingsman1960/sentiscope",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=['sentiscope'],

)