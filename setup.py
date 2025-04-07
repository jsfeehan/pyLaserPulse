import os

from setuptools import find_packages
from setuptools import setup

VERSION = '0.0.0'

long_description = '''pyLaserPulse is a comprehensive simulation toolbox for 
    modelling polarization-resolved, on-axis laser pulse propagation through
    nonlinear, dispersive, passive, and active optical fibre assemblies,
    stretchers, and compressors in python.'''

p = os.path.dirname(__file__)
with open(p + '\\requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pyLaserPulse',
    version=VERSION,
    author='James S. Feehan',
    author_email='pylaserpulse@outlook.com',
    url="https://github.com/jsfeehan/pyLaserPulse",
    description='Python module for pulsed fibre laser and amplifier simulations',
    long_description=long_description,
    license='GPLv3',
    install_requires=required,
    packages=find_packages(),
    package_data={
        'pyLaserPulse.data': [
            'components/loss_spectra/*.dat',
            'fibres/cross_sections/*.dat',
            'materials/loss_spectra/*.dat',
            'materials/Raman_profiles/*.dat',
            'materials/reflectivity_spectra/*.dat',
            'materials/Sellmeier_coefficients/*.dat'
            ],
        'pyLaserPulse.single_plot_window':[
            '*.png',
            '*.ui'
            ]
        }
)
