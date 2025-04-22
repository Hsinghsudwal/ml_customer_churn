from setuptools import find_packages, setup
from typing import List

__version__ = "0.1.0"

REPO_NAME = "ml_customer_churn"
AUTHOR_USER_NAME = "Hsinghsudwal"
SRC_REPO = "ml_customer_churn"
AUTHOR_EMAIL = "sudwalh@gmail.com"


def get_requirements() -> List[str]:
    """Read and return the list of requirements from requirements.txt"""
    try:
        with open("requirements.txt", "r") as file:
            return [
                line.strip() for line in file if line.strip() and line.strip() != "-e ."
            ]
    except FileNotFoundError:
        print("requirements.txt file not found")
        return []


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="End to End Machine Learning Pipeline with MLOps Tools",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
    license="MIT",
)
