"""
Author(s): Matthew Loper

"""

from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
from runpy import run_path

install_reqs = parse_requirements('requirements.txt', session=False)
try:  # for pip < 20.1
    install_requires = [str(ir.req) for ir in install_reqs]
except AttributeError:  # for pip >= 20.1
    install_requires = [str(ir.requirement) for ir in install_reqs]

def get_version():
    namespace = run_path('chumpy/version.py')
    return namespace['version']

setup(name='chumpy',
    version=get_version(),
    packages = ['chumpy'],
    author='Matthew Loper',
    author_email='matt.loper@gmail.com',
    url='https://github.com/mattloper/chumpy',
    description='chumpy',
    install_requires=install_requires,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
)

