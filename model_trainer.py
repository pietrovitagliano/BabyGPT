import os
import sys
import random

import numpy as np
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

from Model.transformer import Transformer, plot_training_history
from utils import apply_gpu_optimizations, encode_text, gpu_check, decode_sequence, fit_encoder_to_text

# FILE NAMES
INPUT_FILE_NAME: str = 'input.txt'
MODEL_FILE_NAME: str = 'baby_gpt.keras'

# GENERAL HYPERPARAMETERS
EPOCH_NUMBER: int = 150
SENTENCE_PER_BATCH_NUMBER: int = 64
SEQUENCE_LENGTH: int = 256
TRAIN_SET_SIZE: float = 0.90

# MODEL HYPERPARAMETERS
LEARNING_RATE: float = 5e-4
D_MODEL: int = 512
POSITIONAL_CONTEXT_SIZE: int = 2 * D_MODEL
HEAD_NUMBER: int = 8
FFN_HIDDEN_LAYER_MULTIPLIER_SIZE: int = 4
DECODER_LAYER_NUMBER: int = 6
DROPOUT: float = 0.25


def get_input_text(input_file_name: str) -> str:
    """
    Read the input text file and return its content.
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


def create_dataset_from_data(data: list[int],
                             sequence_length: int,
                             train_set_size: float,
                             sentence_per_batch_number: int) -> tuple[Dataset, Dataset]:
    """
    Preprocess the input chars and create the training and validation datasets.
    :return: The training and validation datasets
    """

    dataset_size: int = len(data) // sequence_length

    # Initialize the training and validation datasets as empty numpy arrays
    # (uint8 is enough for the ASCII characters and good for memory efficiency)
    x = np.empty(shape=(dataset_size, sequence_length), dtype=np.uint8)
    y = np.empty(shape=(dataset_size, sequence_length), dtype=np.uint8)

    # Fill the training and validation datasets with input and output sequences
    for i in range(dataset_size):
        # x[i] is an input sequence and y[i] is its corresponding output sequence,
        # where each element of y[i] is the next character in the x[i] sequence.
        # That means y[i][j] = x[i][j + 1]
        sentence_start_idx: int = sequence_length * i
        x[i] = data[sentence_start_idx:sentence_start_idx + sequence_length]
        y[i] = data[sentence_start_idx + 1:sentence_start_idx + sequence_length + 1]

    # Turn the training and validation datasets into TensorFlow datasets
    train_set_length = int(train_set_size * dataset_size)
    x_train = x[:train_set_length]
    y_train = y[:train_set_length]
    x_val = x[train_set_length:]
    y_val = y[train_set_length:]

    train_ds = (Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(x_train.shape[0])
                .batch(sentence_per_batch_number, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    test_ds = (Dataset.from_tensor_slices((x_val, y_val))
               .shuffle(x_val.shape[0])
               .batch(sentence_per_batch_number, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE))

    return train_ds, test_ds


def print_random_sample_from_dataset(dataset: Dataset, max_length: int = 15):
    """
    Select a random index from the dataset, print the corresponding input sequence `x` and output sequence `y`,
    and decode them from numbers to characters. The dataset is assumed to be in the format (batch, sentences, chars).

    :param dataset: The dataset (can be training or validation), preprocessed
    :param max_length: The maximum length of the decoded sequences to print
    """

    # Select a random batch index
    random_batch_idx = random.randint(0, len(dataset) - 1)

    # Convert the dataset to a list to access random elements
    dataset_as_list = list(dataset)

    # Extract the random batch (x_sentences, y_sentences)
    x_sentences, y_sentences = dataset_as_list[random_batch_idx]

    # Select a random sentence within the batch
    random_sentence_idx = random.randint(0, len(x_sentences) - 1)

    # Extract the (x, y) pair for the selected sentence
    x = x_sentences[random_sentence_idx].numpy()
    y = y_sentences[random_sentence_idx].numpy()

    # Truncate the sequences if they are longer than the maximum length
    if len(x) > max_length:
        x = x[:max_length]
        y = y[:max_length]

    # Decode the sequences from numbers to characters
    decoded_x = decode_sequence(x)
    decoded_y = decode_sequence(y)

    # Print the decoded sequences
    print(f"X sequence: {decoded_x}"
          f"\n"
          f"Y sequence: {decoded_y}"
          f"\n")


if __name__ == '__main__':
    # If no GPU is available, print a warning message
    if not gpu_check():
        print('No GPU available, the CPU will be used instead.')

    # Apply GPU optimizations (memory growth)
    apply_gpu_optimizations()

    # Read the input text
    input_text: str = get_input_text(input_file_name=INPUT_FILE_NAME)

    # List of all possible characters
    char_list: list = sorted(list(set(input_text)))

    # Fit the encoder to the input text
    fit_encoder_to_text(input_text)

    # Get the absolute path of the model file
    model_abs_path: str = os.path.join(os.getcwd(), MODEL_FILE_NAME)

    # Load the model if it exists, otherwise create a new model
    if os.path.exists(model_abs_path):
        transformer = Transformer.load_model(model_abs_path)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Model {os.path.relpath(model_abs_path, os.path.dirname(project_dir))} loaded successfully.\n")
    else:
        transformer = Transformer(vocab_size=len(char_list),
                                  pos_context_size=POSITIONAL_CONTEXT_SIZE,
                                  decoder_layer_number=DECODER_LAYER_NUMBER,
                                  d_model=D_MODEL,
                                  head_number=HEAD_NUMBER,
                                  dropout_rate=DROPOUT,
                                  ffn_hidden_layer_multiplier=FFN_HIDDEN_LAYER_MULTIPLIER_SIZE)

    # Print a summary of the model and compile it
    transformer.summary(expand_nested=True)
    transformer.compile(optimizer=AdamW(learning_rate=LEARNING_RATE),
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # Get the training and validation batches of the dataset, from the encoded input text
    encoded_input_text: list[int] = encode_text(input_text)
    train_dataset, val_dataset = create_dataset_from_data(data=encoded_input_text,
                                                          sequence_length=SEQUENCE_LENGTH,
                                                          train_set_size=TRAIN_SET_SIZE,
                                                          sentence_per_batch_number=SENTENCE_PER_BATCH_NUMBER)

    # Test a random sample from the training dataset
    print_random_sample_from_dataset(dataset=train_dataset)

    # Define a checkpoint callback to save the best model only
    checkpoint_callback = ModelCheckpoint(
        filepath=model_abs_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = transformer.fit(train_dataset,
                              validation_data=val_dataset,
                              epochs=EPOCH_NUMBER,
                              callbacks=[checkpoint_callback])

    # Plot the training history
    plot_training_history(history)

    # Exit the program
    sys.exit(0)
