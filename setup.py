from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    'xgboost>=0.90',
    'catboost>=0.26',
    'bayesian-optimization>=1.2.0',
    'numpy>=1.19.5',
    'pandas>=1.1.5',
    'matplotlib>=3.2.2',
    'seaborn>=0.11.1',
    'plotly>=4.4.1',
    'pyyaml>=5.4.1'
]

setup(
    name="bonsai-tree",
    version="1.2",
    author="Landon Buechner",
    author_email="mechior.magi@gmail.com",
    description="Bayesian Optimization + Gradient Boosted Trees",
    long_description=readme,
    url="https://github.com/magi-1/bonsai",
    packages=find_packages(),
    package_data={'': ['*.yml']},
    install_requires=requirements,
    license = 'MIT',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)