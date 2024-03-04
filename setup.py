"""Timescale-methods setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('eigvec', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'eigvec',
    version = __version__,
    description = 'Oscillatory eigenvectors.',
    long_description = long_description,
    python_requires = '>=3.6',
    author = 'Ryan Hammonds',
    author_email = 'rphammonds@ucsd.edu',
    maintainer = 'Ryan Hammonds',
    maintainer_email = 'rphammonds@ucsd.edu',
    url = 'https://github.com/voytekresearch/timescale-methods',
    packages = find_packages(),
    #package_dir={'': 'timescales'},
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    platforms = 'any',
    project_urls = {
        #'Documentation' : 'TBD',
        'Bug Reports' : 'https://github.com/voytekresearch/oscillatory_eigenvectors/issues',
        'Source' : 'https://github.com/voytekresearch/oscillatory_eigenvectors'
    },
    download_url = 'https://github.com/voytekresearch/oscillatory_eigenvectors/releases',
    keywords = ['signals', 'time series', 'oscillations', 'eigendecomposition', 'circulant', 'neuroscience'],
    install_requires = install_requires,
    tests_require = ['pytest', 'pytest-cov'],
    extras_require = {
        'tests'   : ['pytest', 'pytest-cov'],
        'all'     : ['pytest', 'pytest-cov']
    }
)
