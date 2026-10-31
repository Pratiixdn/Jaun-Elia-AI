
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./jaun_elia_model")

print("Transformer layers:", len(model.transformer.h))
print("\nFirst Transformer layer:")
print(model.transformer.h[0].attn)

print("\nNumber of attention heads:")
print(model.config.n_head)

print("\nHidden size:")
print(model.config.n_embd)
