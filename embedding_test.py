import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./jaun_elia_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

text = "Shayad main tumhara hoon"

# Convert text into token IDs
inputs = tokenizer(text, return_tensors="pt")

# Get the token IDs
token_ids = inputs["input_ids"]

# Get the embeddings
with torch.no_grad():
    embeddings = model.transformer.wte(token_ids)

print("Token IDs:")
print(token_ids)

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nFirst token's first 10 embedding values:")
print(embeddings[0, 0, :10])