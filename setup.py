import setuptools

setuptools.setup(name='bpnsdata',
                 version='0.1.11',
                 description='Add Belgian Part of the North Sea (BPNS) marine environmental data to a geodataframe',
                 author='Clea Parcerisas',
                 author_email='clea.parcerisas@vliz.be',
                 url="https://github.com/lifewatch/bpnsdata.git",
                 license='',
                 packages=setuptools.find_packages(),
                 install_requires=['owslib', 'pygeos', 'tqdm', 'erddapy', 'rioxarray', 'skyfield', 'netcdf4',
                                   'contextily', 'geopandas'],
                 extras_require={
                     "time": ["skyfield"],
                     "griddap": ["erddapy", 'xarray']
                 },
                 package_data={
                                "bpnsdata": ["data/*.*"]
                              },
                 zip_safe=False)
