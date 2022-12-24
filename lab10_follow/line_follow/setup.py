from setuptools import setup

package_name = 'line_follow'

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
    maintainer='dhayakar',
    maintainer_email='dhayakar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_spot = line_follow.ground_spot:main',
            'pure_pursuit = line_follow.pure_pursuit:main',
            'pid_run = line_follow.pid_run:main',
        ],
    },
)
