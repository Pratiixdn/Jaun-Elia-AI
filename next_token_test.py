
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./jaun_elia_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

text = "Zindagi"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get predictions for the next token
logits = outputs.logits[0, -1, :]

# Convert scores into probabilities
probabilities = torch.softmax(logits, dim=-1)

# Find the 10 most likely next tokens
top_probabilities, top_ids = torch.topk(probabilities, 10)

print("Prompt:", text)
print("\nTop 10 next-token predictions:\n")

for token_id, probability in zip(top_ids, top_probabilities):
    token = tokenizer.decode([token_id.item()])
    print(f"{token!r:20} {probability.item():.4%}")
