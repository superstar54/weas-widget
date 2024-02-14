from setuptools import setup, find_packages

setup(
    name="weas-widget",
    version="0.0.4",
    packages=find_packages(),
    description="A widget to visualize and interact with atomistic structures in Jupyter Notebook.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Xing Wang",
    author_email="xingwang1991@gmail.com",
    url="https://github.com/superstar54/weas-widget",
    install_requires=[
        "anywidget",
        "ipywidgets",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        "": ["*.js", "*.css"],
        "weas_widget.datas": ["*"],
    },
    include_package_data=True,  # This tells setuptools to check MANIFEST.in for additional files
)
