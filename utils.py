import os

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

input_text_encoder: LabelEncoder = LabelEncoder()


def gpu_check() -> bool:
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Return if there are available GPUs
    return len(gpus) > 0


def apply_gpu_optimizations():
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Avoid Out of Memory errors by enabling, for any gpu,
    # the GPU Memory Consumption Growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def encode_text(text: str) -> list[int]:
    """
    Encode the given string as a sequence of integers, using the given label encoder.
    :param text: The string to encode
    :return: The encoded sequence of integers
    """
    global input_text_encoder

    return input_text_encoder.transform(list(text))


def decode_sequence(encoded_sequence: list[int]) -> str:
    """
    Decode the encoded sequence of integers provided as a string, using the given label encoder.
    :param encoded_sequence: The encoded sequence of integers to decode
    :return: The decoded text
    """
    global input_text_encoder

    return ''.join(input_text_encoder.inverse_transform(encoded_sequence))


def get_input_text(input_file_name: str) -> str:
    """
    Read the input text file and return the text content.
    :return: str: The text content of the input file
    """

    # Get the absolute path of the input file and check if it exists
    input_file_abs_path = os.path.join(os.getcwd(), input_file_name)
    if not os.path.exists(input_file_abs_path):
        raise FileNotFoundError(f"Input file '{input_file_name}' not found.")

    # Read the input text file
    with open(input_file_abs_path, 'r', encoding='utf-8') as input_file:
        text = input_file.read()

    return text


def fit_encoder_to_text(input_text: str):
    global input_text_encoder

    # Get the sorted list of all possible characters
    chars = sorted(list(set(input_text)))

    input_text_encoder.fit(chars)
