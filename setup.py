from setuptools import setup, find_packages

setup(
    name='solarimage',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=False,
    url='',
    license='',
    author='SolarFresh',
    author_email='shangyuhuang@gmail.com',
    description='',
    install_requires=[
        "opencv-python==3.2.0.7",
        "pandas==0.20.3",
        "plotly==2.0.12",
        "tensorflow==1.2.1",
    ]
)
