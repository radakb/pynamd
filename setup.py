from setuptools import setup
from setuptools.command.build_py import build_py

if __name__ == '__main__':
    setup(name = 'pynamd',
          version = '1.0',
          description = 'Python Tools for NAMD',
          author = 'Brian K. Radak',
          license = 'MIT',
          packages = ['pynamd', 'pynamd.msmle'],
          cmdclass = {'build_py': build_py},
         )
