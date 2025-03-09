import os

from setuptools import setup


def __parse_requirements(file_name):
    with open(file_name, 'r') as f:
        line_striped = (line.strip() for line in f.readlines())
    requirements = [line for line in line_striped if line and not line.startswith('#')]
    # requirements.append(__check_pytorch())
    return requirements


# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='py3dv',
    author='Qinfeng Xiao',
    author_email='qin-feng.xiao@connect.polyu.hk',
    description='A python package for 3d shape analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.0.1a1',
    packages=[],
    scripts=[],
    install_requirements=__parse_requirements('requirements.txt'),
    install_requires=__parse_requirements('requirements.txt'),
    # cmdclass={'install': __PostInstallMoveFile},
    url='https://github.com/larryshaw0079/Corr3D',
    python_requires=">=3.6",
    license='GPL-3.0 License'
)
