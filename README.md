# 🧠 BabyGPT From Scratch – Decoder-Only Transformer in PyTorch

This project is a **minimal GPT-style decoder-only Transformer**, built completely from scratch using PyTorch. It is inspired by [Andrej Karpathy's](https://github.com/karpathy) excellent work on nanoGPT. The goal was to understand the internal mechanics of transformer architectures by re-implementing every component manually, while keeping the setup light enough to run even on local machines with low compute.

---

## 🚀 Introduction

Inspired by Andrej Karpathy’s educational walkthroughs, I started building a **GPT-style Transformer decoder** from scratch. This project implements a **decoder-only language model** trained on a character-level Tiny Shakespeare dataset.

The goal was to:
- Understand the core components of Transformer architectures
- Learn how multi-head self-attention and feedforward networks work
- Train a functioning model in low-resource environments
- Leave hooks for **scaling up** depending on your available FLOPs (GPU/TPU/CPU)

---

## 🧱 Model Architecture

The model implements a **Transformer decoder stack** consisting of:

- Token Embedding Layer
- Positional Embedding Layer
- N repeated Transformer Blocks:
  - Multi-head causal self-attention
  - Feedforward MLP
  - Residual connections
  - LayerNorm
- Final linear projection head

> **Causal masking** is applied to ensure that the model cannot peek ahead during training.


---

## ⚙️ Setup

### Requirements:
- Python 3.7+
- PyTorch (`pip install torch`)
- VS Code (recommended)

### Dataset:
Download Tiny Shakespeare:

wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
.
├── input.txt             # Tiny Shakespeare dataset
├── main.py               # Full GPT-like training + generation code
├── README.md             # You're reading it!
├── DETAILS.md            # Author, license, references


batch_size = 32       # sequences per batch
block_size = 8        # context length
n_embd = 96           # embedding dimensions
n_head = 3            # number of attention heads






---





# 📜 Project Details & Credits

## 👨‍💻 Author

Made with ❤️ by **Adithya Jayasundar TS**  
A learning project to build Transformer models from scratch and gain deep understanding.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📚 References

- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [GPT: Improving Language Understanding by Generative Pretraining](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Transformer Illustrated](https://jalammar.github.io/illustrated-transformer/)

n_layer = 3           # transformer blocks
dropout = 0.2         # dropout rate
learning_rate = 3e-2  # initial learning rate

