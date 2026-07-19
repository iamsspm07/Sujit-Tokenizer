# Sujit-Tokenizer

A custom Byte-Level BPE (Byte Pair Encoding) Tokenizer implemented from scratch in Python.

## Features

* Byte-level tokenization
* Custom BPE training
* Text encoding and decoding
* Save and load tokenizer models
* UTF-8 support
* Lightweight and dependency-free

## Installation

### Clone Repository

```bash
git clone https://github.com/iamsspm07/Sujit-Tokenizer.git
cd Sujit-Tokenizer
```

### Install Package

```bash
pip install -e .
```

## Project Structure

```text
Sujit-Tokenizer/
│
├── Sujit_Tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
│
├── corpus.txt
├── train_and_test.py
│
├── README.md
├── setup.py
├── pyproject.toml
└── LICENSE
```

## Quick Start

### Import Tokenizer

```python
from Sujit_Tokenizer import CustomByteLevelBPETokenizer
```

### Train Tokenizer

```python
corpus = [
    "Transformers are amazing.",
    "Machine Learning is powerful.",
    "Python is widely used in AI."
]

tokenizer = CustomByteLevelBPETokenizer(
    vocab_size=1000
)

tokenizer.train(corpus)
```

### Save Model

```python
tokenizer.save_model(
    "tokenizer.model"
)
```

### Load Model

```python
tokenizer = CustomByteLevelBPETokenizer()

tokenizer.load_model(
    "tokenizer.model"
)
```

### Encode Text

```python
encoded = tokenizer.encode(
    "Transformers use attention."
)

print(encoded)
```

Example Output:

```python
[2, 451, 723, 812, 3]
```

### Decode Text

```python
decoded = tokenizer.decode(
    encoded
)

print(decoded)
```

Output:

```text
Transformers use attention.
```

## Training Workflow

```text
Corpus
   ↓
Byte Conversion
   ↓
Pair Frequency Counting
   ↓
BPE Merging
   ↓
Vocabulary Construction
   ↓
Tokenizer Model
```

## Special Tokens

| Token | ID |
| ----- | -- |
| <PAD> | 0  |
| <UNK> | 1  |
| <BOS> | 2  |
| <EOS> | 3  |

## Example

```python
from Sujit_Tokenizer import CustomByteLevelBPETokenizer

tokenizer = CustomByteLevelBPETokenizer(
    vocab_size=1000
)

corpus = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning"
]

tokenizer.train(corpus)

tokenizer.save_model(
    "tokenizer.model"
)

tokenizer.load_model(
    "tokenizer.model"
)

text = "Machine Learning"

encoded = tokenizer.encode(text)
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
```

## Use Cases

* NLP experiments
* Tokenization research
* Educational projects
* Understanding BPE internals
* Building custom language models
* Learning tokenizer architecture

## Future Enhancements

* Faster BPE training
* WordPiece tokenizer
* SentencePiece tokenizer
* Parallel processing
* Vocabulary statistics
* Token frequency analysis
* Hugging Face compatibility

## Author

Sujit Maity, 
Praveen Kumar Kannan

## License

MIT License
