from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'dv_stack'

# Coleta CSVs; se não existir nenhum, NÃO adiciona a pasta
maps_files = glob(os.path.join('maps', '*.csv'))

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
if maps_files:
    data_files.append(('share/' + package_name + '/maps', maps_files))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='pedroromero',
    maintainer_email='galaga100prs@gmail.com',
    description='Planner + fontes fake',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dv_planner_node = dv_stack.dv_planner_node:main',
            'dv_fake_sources = dv_stack.dv_fake_sources:main',
        ],
    },
)
