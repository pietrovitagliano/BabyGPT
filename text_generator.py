import os
import sys

from Model.transformer import Transformer
from utils import (apply_gpu_optimizations, encode_text, gpu_check, decode_sequence,
                   get_input_text, fit_encoder_to_text)

# FILE NAMES
INPUT_FILE_NAME: str = 'input.txt'
MODEL_FILE_NAME: str = 'baby_gpt.keras'

# CONSTANTS
SEQUENCE_LENGTH: int = 100
INPUT_FOR_GENERATION: str = "Hello, "

if __name__ == '__main__':
    # If no GPU is available, print a warning message
    if not gpu_check():
        print('No GPU available, the CPU will be used instead.')

    # Apply GPU optimizations (memory growth)
    apply_gpu_optimizations()

    # Read the input text
    input_text: str = get_input_text(input_file_name=INPUT_FILE_NAME)

    # Fit the encoder to the input text
    fit_encoder_to_text(input_text)

    # Get the absolute path of the model file
    model_abs_path: str = os.path.join(os.getcwd(), MODEL_FILE_NAME)

    # Load the model if it exists, otherwise raise an error
    if not os.path.exists(model_abs_path):
        raise FileNotFoundError(f"Model file '{model_abs_path}' not found.")

    transformer = Transformer.load_model(model_abs_path)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Model {os.path.relpath(model_abs_path, os.path.dirname(project_dir))} loaded successfully.\n")

    # Generate a sequence of tokens from a given input sequence
    encoded_sequence: list[int] = transformer.generate_tokens_from_input(
        sequence_length=SEQUENCE_LENGTH,
        initial_context=encode_text(INPUT_FOR_GENERATION)
    )

    # Print the input and the generated sequences
    print(f"INPUT GIVEN: {INPUT_FOR_GENERATION}\n"
          f"\n"
          f"GENERATED SEQUENCE: {decode_sequence(encoded_sequence)}")

    # Exit the program
    sys.exit(0)
