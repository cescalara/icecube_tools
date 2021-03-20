import setuptools
import versioneer

setuptools.setup(
    name="icecube_tools",
    packages=setuptools.find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
