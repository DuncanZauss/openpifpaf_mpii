from setuptools import setup, find_packages

setup(
    name='openpifpaf_mpii',
    packages= ['openpifpaf_mpii'],
    license='MIT',
    version = '0.1.0',
    description='OpenPifPaf MPII datset PlugIn',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Duncan Zauss',
    url='https://github.com/DuncanZauss/openpifpaf_mpii',

    install_requires=[
        'matplotlib',
        'numpy',
        'openpifpaf>=0.12b1',
    ],
)
