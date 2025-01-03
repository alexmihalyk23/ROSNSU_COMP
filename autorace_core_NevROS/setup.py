from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autorace_core_NevROS'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alexmihalyk23@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'robot_control_node = autorace_core_NevROS.robot_control_node:main',
        'task_action_server = autorace_core_NevROS.task_action_server:main',
        'task_action_client = autorace_core_NevROS.task_action_client:main',
        ],
    },
)
