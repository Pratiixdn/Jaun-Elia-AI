
from transformers import pipeline, AutoTokenizer

MODEL_PATH = "./jaun_elia_model"

print("Loading Jaun Elia AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

poet = pipeline(
    "text-generation",
    model=MODEL_PATH,
    tokenizer=tokenizer
)

def generate_poetry(prompt_text, max_new_tokens=100, temp=0.8):
    output = poet(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.15,
        num_return_sequences=1,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    return output[0]["generated_text"]


if __name__ == "__main__":
    print("\n== Jaun Elia AI Poetry Generator ==")

    user_prompt = input(
        "Enter an opening prompt (e.g., Shayad, Ab main, Tum): "
    ).strip()

    if not user_prompt:
        user_prompt = "Shayad"

    print("\nGenerating poetry...\n")

    poetry = generate_poetry(user_prompt)

    print("--- Result ---")
    print(user_prompt + poetry)
