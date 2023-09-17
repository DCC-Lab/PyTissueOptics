import setuptools

""" 
To distribute:
=============
rm dist/*; python setup.py sdist bdist_wheel; python -m twine upload dist/* 
"""

setuptools.setup(
    name="pytissueoptics",
    version="2.0.0-beta.1",
    url="https://github.com/DCC-Lab/PyTissueOptics",
    authors=["Marc-Andre Vigneault", "Ludovick Begin", "Daniel Cote"],
    authors_emails=["marc-andre.vigneault.2@ulaval.ca", "ludovick.begin.1@ulaval.ca", "dccote@cervo.ulaval.ca"],
    description="Monte Carlo Light Propagation in Python for tissues",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='tissue optics monte carlo light propagation simulation',
    packages=setuptools.find_packages(),
    install_requires=['matplotlib', 'numpy'],
    python_requires='>=3.6',
    package_data={'': ['*.png'], "doc": ['*.html']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Education',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Operating System :: OS Independent'
    ],
)
