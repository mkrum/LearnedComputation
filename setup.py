from distutils.core import setup
import glob

setup(
    name="lc",
    packages=["lc"],
    install_requires=["torch", "rich", "bitstring", "matplotlib", "seaborn"],
)
