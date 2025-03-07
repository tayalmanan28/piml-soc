def read_requirements():
    with open(Path(__file__).parent / "requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="piml_soc",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),  # Use the function to get dependencies
    python_requires=">=3.7",  # Specify Python version compatibility
)
