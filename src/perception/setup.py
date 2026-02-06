from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pedro',
    maintainer_email='galaga100prs@gmail.com',
    description='Pacote de perception com YOLOv5 para cones',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'processing = perception.processing:main',
            'keyboard_control = perception.keyboard_control:main',
            'path_planner = perception.path_planner:main',
            'novo_path = perception.novo_path:main',
            'control = perception.control:main',
            'path_local = perception.path_local:main'
        ],
    },
)

