from setuptools import find_packages
from setuptools import setup

with open("requirements.txt", encoding='utf-8') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='P7_Bad_buzz',
      version="0.0.0",
      description="Détection de Bad Buzz",
      license="MIT",
      author="pjcggf",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
