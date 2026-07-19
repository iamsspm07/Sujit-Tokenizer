from setuptools import setup, find_packages

setup(
    name="Sujit-Tokenizer",
    version="1.0.0",
    author="Sujit Shibaprasad Maity, Praveen Kumar Kannan",
    author_email="sujitmaity.in@gmail.com, Praveenkumar.kannan.gmail.com",
    description="Custom Byte-Level BPE Tokenizer built from scratch in Python",
    long_description=open(
        "README.md",
        encoding="utf-8"
    ).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iamsspm07/Sujit-Tokenizer",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    keywords=[
        "tokenizer",
        "bpe",
        "byte-pair-encoding",
        "nlp",
        "llm",
        "machine-learning",
        "artificial-intelligence"
    ],
)
