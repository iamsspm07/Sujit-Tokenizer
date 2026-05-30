from Sujit_Tokenizer.tokenizer import CustomByteLevelBPETokenizer

# Load corpus
with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = [
        line.strip()
        for line in file
        if line.strip()
    ]

print("=" * 60)
print("SUJIT TOKENIZER")
print("=" * 60)

print(f"Loaded {len(corpus)} training sentences")

# Create tokenizer
tokenizer = CustomByteLevelBPETokenizer(
    vocab_size=1000
)

# Train tokenizer
print("\nTraining tokenizer...\n")
tokenizer.train(corpus)

# Save trained tokenizer
tokenizer.save_model(
    "tokenizer.model"
)

# Load tokenizer
loaded_tokenizer = CustomByteLevelBPETokenizer()
loaded_tokenizer.load_model(
    "tokenizer.model"
)

# Test text
text = "Transformers use attention."

print("\n" + "=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(text)

# Encode
encoded = loaded_tokenizer.encode(text)

print("\n" + "=" * 60)
print("ENCODED TOKEN IDS")
print("=" * 60)
print(encoded)

# Decode
decoded = loaded_tokenizer.decode(encoded)

print("\n" + "=" * 60)
print("DECODED TEXT")
print("=" * 60)
print(decoded)

# Verify
if text == decoded:
    print("\n✓ Encode/Decode Successful")
else:
    print("\n✗ Encode/Decode Failed")

print("\n" + "=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)