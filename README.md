# BabyGPT: A Transformer from Scratch

**BabyGPT** is a Python project that implements a Transformer model from the ground up using TensorFlow and is based on the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper.

This project demonstrates how a Transformer can be constructed to learn and generate text based on a given input corpus, mimicking the style of the provided dataset.

## Features

- Full implementation of the Transformer architecture from scratch.
- Capable of text generation by learning from a custom text file.
- Easy to train using TensorFlow.
- Well-structured codebase to facilitate understanding of the Transformer architecture.

## Model Overview

The Transformer model adheres to the architecture outlined in the paper, customizing it for the text generation, which includes:

- **Token Embedding**
- **Positional Encoding**
- **Multi-Head Self-Attention Mechanism**
- **Decoder Stack**

The model can be trained on a text input file, learning to generate coherent text based on the patterns and structure observed in the data.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pietrovitagliano/BabyGPT.git

2. **Install the dependencies:**
   ```bash
    pip install -r requirements.txt
   
3. **Train the model:**
   Run the `model_trainer.py` script to train and save the model. This script is designed to load an existing model if available, allowing for continued training across multiple sessions without losing previous progress.

4. **Text Generation:**
   Use the `text_generator.py` script to load the trained model and use it. Simply provide a starting phrase and the model will generate text in the learned style.
   
## Licence
This project is licensed under the MIT License. See the LICENSE file for more details.
