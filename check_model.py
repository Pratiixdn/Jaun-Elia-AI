
from transformers import AutoModelForCausalLM

model_path = "./jaun_elia_model"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(model_path)

# Total parameters
total_params = sum(p.numel() for p in model.parameters())

# Trainable parameters
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("\n===== MODEL INFORMATION =====")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Parameters in millions: {total_params / 1_000_000:.2f}M")
print(f"Model architecture: {model.config.model_type}")
print(f"Layers: {model.config.n_layer}")
print(f"Vocabulary size: {model.config.vocab_size}")
print(f"Hidden size: {model.config.n_embd}")
