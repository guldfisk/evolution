from setuptools import setup
import os

def package_files(directory):
    paths = []
    for path, directories, file_names in os.walk(directory):
        for filename in file_names:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('evolution')

setup(
    name = 'evolution',
    version = '1.0',
    packages = ['evolution'],
    package_data = {'': extra_files},
    install_requires = [
        'numpy',
    ],
)