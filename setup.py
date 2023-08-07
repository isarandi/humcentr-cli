from setuptools import setup

setup(
    name='humcentr-cli',
    version='0.1.1',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['humcentr_cli'],
    license='LICENSE',
    description='Command-line interface for various human-centric computer vision tasks (person '
                'detection, segmentation, 3D pose estimation)',
    python_requires='>=3.8',
    install_requires=[
        'tensorflow',
        'opencv-python',
        'imageio',
        'more-itertools',
        'simplepyutils @ git+https://github.com/isarandi/simplepyutils.git',
        'cameralib @ git+https://github.com/isarandi/cameralib.git',
        'poseviz @ git+https://github.com/isarandi/poseviz.git',
    ],
)
