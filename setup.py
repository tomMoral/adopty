#! /usr/bin/env python
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup
from pip.req import parse_requirements

descr = """Adaptive Optimization for the LASSO"""

DISTNAME = 'adopty'
DESCRIPTION = descr
MAINTAINER = 'Thomas Moreau'
MAINTAINER_EMAIL = 'thomas.moreau.2010@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/tommoral/adopty.git'
VERSION = '0.1.dev0'


# parse_requirements() returns generator of pip.req.InstallRequirement objects
with open('requirements.txt') as f:
    REQUIREMENTS = [r.strip() for r in f.readlines()]
REQUIREMENTS = [r for r in REQUIREMENTS if r != '']

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'adopty'
          ],
          install_requires=REQUIREMENTS
          )
