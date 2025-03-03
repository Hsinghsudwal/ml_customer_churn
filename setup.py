from setuptools import find_packages, setup
from typing import List


__version__ = "0.0.1"

REPO_NAME = "ml_customer_churn"
AUTHOR_USER_NAME = "Hsinghsudwal"
SRC_REPO = "ml_customer_churn"
AUTHOR_EMAIL = "sudwalh@gmail.com"

def get_requirements()->List[str]:
    
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='End to End Machine learning pipeline with MLOps tools',
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=find_packages(),
    install_requires=get_requirements(),
    license='MIT',
)

