from distutils.core import setup
from distutils.command.build_py import build_py

packages = ['pynamd']

if __name__ == '__main__':
    setup(name = 'pynamd',
          version = '1.0',
          description = 'Python Tools for NAMD',
          author = 'Brian K. Radak',
          license = 'MIT',
          packages = packages,
          cmdclass = {'build_py': build_py},
         )
