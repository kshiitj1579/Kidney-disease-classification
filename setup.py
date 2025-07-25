#file used in Python projects for packaging and distribution
import setuptools
#This line imports the setuptools library, which is the primary tool for packaging Python projects.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
#read the read.me and prinmts everything on the site to explain it

__version__ = "0.0.0"
#very inital phase of the project

REPO_NAME = "Kidney-disease-classification"
AUTHOR_USER_NAME = "kshiitij1579"
SRC_REPO = "med_classifier" #src k andar jo folder h
AUTHOR_EMAIL = "kshitij.bhatnagar1579@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)