from setuptools import find_packages, setup

setup(
    name='adsiat2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='To predict beer type based on other metrics',
    author='GROUP_3',
    license='',
)
