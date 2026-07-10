from setuptools import setup, find_packages

setup(
    name='automatic-ar',
    version='1.0.0',
    description='Python port of automatic-ar: simultaneous multi-view camera pose estimation '
                'and object tracking with squared planar markers.',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19',
        'opencv-contrib-python>=4.5',
        'scipy>=1.7',
        'pyyaml>=5.4',
    ],
    entry_points={
        'console_scripts': [
            'detect_markers=apps.detect_markers:main',
            'find_solution=apps.find_solution:main',
            'overlay=apps.overlay:main',
            'track=apps.track:main',
        ],
    },
    python_requires='>=3.8',
)
