## What I'm using

- **Python** — for the entire project
- **PyTorch** — for the underlying machine learning operations
- **Hugging Face Transformers** — to load, train, and use the language model
- **DistilGPT-2** — the pretrained model I fine-tuned
- **PowerShell and a Python virtual environment** — for development on Windows

## How it works

I started with a pretrained DistilGPT-2 model and fine-tuned it on a small text dataset containing Roman Urdu poetry attributed to Jaun Elia.

After training, I used the saved model to generate text from a prompt. For example, I can give it a word or a short phrase and let it predict what comes next.

I also wrote small scripts to experiment with different parts of the model, including:

- Tokenization: how text is split into smaller pieces.
- Embeddings: how tokens are represented as numerical vectors.
- Next-token prediction: how the model chooses what text might come next.
- Self-attention: how the model uses information from earlier tokens.

These experiments helped me understand the basics of how GPT-style language models process text.

## Project structure

```text
my_personal_ai/
├── data/
│   └── jaun_elia_data.txt
├── jaun_elia_model/
├── train.py
├── generate.py
├── check_model.py
├── tokenizer_test.py
├── embedding_test.py
├── next_token_test.py
├── attention_test.py
├── view_attention.py
├── requirements.txt
└── README.md
```

The `jaun_elia_model/` directory contains the saved model and tokenizer locally. Depending on how the project is published, these files may be stored separately rather than committed to Git.

## Getting started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd my_personal_ai
```

Replace `YOUR_REPOSITORY_URL` with the URL of this repository.

### 2. Create a virtual environment

On Windows, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Generate poetry

If the trained model is available locally, run:

```powershell
python generate.py
```

If you want to train the model yourself, make sure the dataset is available and check the settings in `train.py` before starting. Training can take a while, especially on a CPU.

**Note:** The commands above assume that the repository contains the required scripts and dependencies. If the model weights or dataset are stored separately, download or prepare them before running the generator.

## Current limitations

This is still an experimental project, and there are several things I need to improve.

- The dataset is small, which limits the variety and consistency of the generated text.
- DistilGPT-2 was pretrained primarily on English text, so it isn't specifically optimized for Roman Urdu.
- Some generated passages are grammatically incorrect or don't make much sense.
- Roman Urdu spelling varies, which can make learning consistent patterns harder.
- The quality of the generated poetry hasn't been evaluated systematically.

Fine-tuning a model doesn't automatically make it a good poet. The quality of the data, the pretrained model, and the training process all matter.

## What's next?

Here are some things I'd like to work on as I continue learning:

- Clean up and organize the training dataset.
- Find additional poetry text that I have permission to use.
- Experiment with models that are better suited to multilingual or Roman Urdu text.
- Compare different training settings instead of simply increasing the number of epochs.
- Evaluate generated samples more carefully.
- Improve the generation script and make the project easier to run.
- Eventually, build a simple interface for generating poetry.

## Why I built this

I'm interested in AI and machine learning, and I wanted to go beyond simply using AI tools. I wanted to understand how a language model can be trained, how it represents text, and why it sometimes generates useful text and sometimes produces nonsense.

This project is one step in that learning process. It isn't a finished AI product, but it's a practical way for me to learn by building something myself.

---

**Status:** Work in progress.

If you have suggestions for improving the model, the dataset, or the training process, feel free to open an issue or share your ideas.
