# BabyGPT – Decoder-Only Transformer in PyTorch

A minimal implementation of a **GPT-style decoder-only Transformer**, built from scratch using **PyTorch**.  
This project focuses on understanding the internal mechanics of transformer architectures by implementing each component manually while keeping the model lightweight enough to run on modest hardware.

The model is trained on the **Tiny Shakespeare** dataset at the **character level**.

---

## Overview

This project was developed as a learning exercise to explore how modern language models work internally. Instead of relying on high-level libraries, the core transformer components are implemented directly in PyTorch.

Objectives of the project:

- Understand the architecture of GPT-style language models  
- Implement **multi-head self-attention** from scratch  
- Build a **decoder-only Transformer stack**
- Train a functional language model in a **low-resource environment**
- Provide a simple base that can later be **scaled to larger models**

---

## Model Architecture

The model follows the standard **Transformer decoder architecture**.

Main components:

- **Token Embedding Layer** – converts characters into embedding vectors  
- **Positional Embedding Layer** – encodes positional information of tokens  
- **Transformer Blocks** (repeated N times):
  - Multi-Head **Causal Self-Attention**
  - Feedforward **MLP**
  - **Residual Connections**
  - **Layer Normalization**
- **Linear projection head** for next-token prediction

A **causal attention mask** is applied to ensure the model only attends to previous tokens during training.

---

## Dataset

The model is trained on the **Tiny Shakespeare dataset**, a commonly used corpus for character-level language modeling experiments.

Download the dataset:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
