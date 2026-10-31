
from transformers import AutoTokenizer

# Load your trained model's tokenizer
tokenizer = AutoTokenizer.from_pretrained("./jaun_elia_model")

text = "Shayad main tumhara hoon"

# Convert text into tokens
tokens = tokenizer.tokenize(text)

# Convert tokens into numerical IDs
token_ids = tokenizer.encode(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))

# Convert IDs back into text
decoded_text = tokenizer.decode(token_ids)

print("Decoded text:", decoded_text)
words = [
    "Mohabbat",
    "Ishq",
    "Zindagi",
    "Tanhai",
    "Hijr",
    "Khud-kushi",
    "Jaun Elia"
]

print("\n===== ROMAN URDU TOKENIZATION =====")

for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"{word:15} -> {tokens} -> {len(tokens)} tokens")
    