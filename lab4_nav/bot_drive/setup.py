from setuptools import setup

package_name = 'bot_drive'

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
    maintainer_email='dhayakar@msu.edu',
    description='Turtle bot move package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'circle_drive = bot_drive.circle_drive:main',
				'bot_monitor = bot_drive.bot_monitor:main',
				'square_drive = bot_drive.square_drive:main',
			
        ],
    },
)
