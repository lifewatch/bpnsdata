import setuptools

setuptools.setup(name='bpnsdata',
                 version='0.1',
                 description='Add BPNS sea data to a geodataframe',
                 author='Clea Parcerisas',
                 author_email='clea.parcerisas@vliz.be',
                 url="/",
                 license='',
                 packages=setuptools.find_packages(),
                 package_data={
                                "bpnsdata": ["data/*.*"]
                              },
                 include_package_data=True,
                 zip_safe=False)