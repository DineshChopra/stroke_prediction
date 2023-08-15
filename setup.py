from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str = 'requirements.txt') -> list:
    with open(file_path) as f:
      # return f.read().splitlines()
      requirements = f.read().splitlines()
      return [requirement for requirement in requirements if HYPHEN_E_DOT not in requirement]

setup(
  name='stroke_prediction',
  version='0.1.0',
  author='Dinesh Chopra',
  author_email='XXXXXXXXXXXXXXXXXXXXXXXX',
  description='Stroke Prediction End to End Project',
  packages=find_packages(),
  install_requires=get_requirements(),
  license='MIT',
)
