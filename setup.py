import setuptools

setuptools.setup(name='soundexplorer',
                 version='0.1',
                 description='Create acoustic datasets and explore the outcome for large deployments.',
                 author='Clea Parcerisas',
                 author_email='clea.parcerisas@vliz.be',
                 url="/",
                 license='',
                 packages=setuptools.find_packages(),
                 package_data={
                                "soundexplorer": ["data/de421.bsp"]
                              },
                 zip_safe=False)