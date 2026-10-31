
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./jaun_elia_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager"
)

model.eval()

text = "Main tumse mohabbat karta hoon"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        return_dict=True
    )

if outputs.attentions is None or len(outputs.attentions) == 0:
    raise RuntimeError(
        "Attention weights were not returned. Check the model configuration."
    )

# First layer, first attention head
attention = outputs.attentions[0]
first_head = attention[0, 0]

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("Tokens:", tokens)
print("\nAttention weights from the first head:")

for i, token in enumerate(tokens):
    print(f"\nToken: {token!r}")

    for j, previous_token in enumerate(tokens):
        weight = first_head[i, j].item()
        print(f"  {previous_token!r}: {weight:.4f}")
