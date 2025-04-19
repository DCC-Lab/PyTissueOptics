import setuptools

""" 
To distribute:
=============
rm dist/*; python setup.py sdist bdist_wheel; python -m twine upload dist/* 
"""

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="pytissueoptics",
    version="2.0.0b1",
    url="https://github.com/DCC-Lab/PyTissueOptics",
    author=["Ludovick Begin", "Marc-Andre Vigneault", "Daniel Cote"],
    author_email=["ludovick.begin@gmail.com", "marc-andre.vigneault.2@ulaval.ca", "dccote@cervo.ulaval.ca"],
    description="Python module for 3D Monte Carlo Simulation of Light Propagation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='tissue, optics, monte carlo, light propagation, simulation, scattering, ray tracing, 3D',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'mockito',
        'psutil',
        'pyopencl',
        'pyqt5',
        'configobj',
        'traitsui',
        'vtk>=9.2.2',
        'mayavi>=4.8.2',
        'Pygments',
    ],
    python_requires='>=3.9',
    package_data={'pytissueoptics': ['rayscattering/opencl/src/*.c', '**/*.obj', 'examples/*.py']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent'
    ],
)
