import os

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LayerNormalization, Dense, Softmax
from tensorflow.keras.callbacks import History

from matplotlib import pyplot as plt

from Model.decoder import Decoder
from Model.multi_head_attention import MultiHeadAttention, HeadAttention


def plot_training_history(history: History):
    """
    Plot the training history of a model, taking into account the loss and the accuracy.
    :param history: The training history of a model
    """

    _, ax = plt.subplots(1, 2, figsize=(12, 10))

    # Plot the training loss and the validation loss
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    # Plot the training accuracy and the validation accuracy
    ax[1].plot(history.history['accuracy'], label='train_accuracy')
    ax[1].plot(history.history['val_accuracy'], label='val_accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


class Transformer(Model):
    def __init__(self, vocab_size: int, pos_context_size: int, decoder_layer_number: int, d_model: int,
                 head_number: int, dropout_rate: float, ffn_hidden_layer_multiplier: int, **kwargs):
        super().__init__(**kwargs)

        self._embedding_layer = Embedding(vocab_size, d_model)
        self._pos_embedding_layer = Embedding(pos_context_size, d_model)

        decoder_sequence = Sequential([
            Decoder(d_model=d_model,
                    head_number=head_number,
                    dropout_rate=dropout_rate,
                    ffn_hidden_layer_multiplier=ffn_hidden_layer_multiplier)
            for _ in range(decoder_layer_number)
        ])

        self._model = Sequential([
            decoder_sequence,
            LayerNormalization(),
            Dense(vocab_size),
            Softmax(axis=-1)
        ])

        self._model.build(input_shape=(None, vocab_size, d_model))

    def call(self, x_batch, *args, **kwargs):
        """
        Process the input batch through the transformer model and return the output
        :param x_batch: The batch of input sequences (NOTE: x_batch must contain only the input without the labels)
        :param args: The additional arguments (optional)
        :param kwargs: The additional keyword arguments (optional)
        :return: A 2-D tensor of the same shape of x_batch (batch size x number of token) with the probabilities
        for each token in the vocabulary
        """

        # Create embeddings matrix
        embeddings = self._embedding_layer(x_batch)

        # Create positional embeddings matrix
        token_sequence_length: int = x_batch.shape[1]
        pos_embeddings = self._pos_embedding_layer(tf.range(token_sequence_length))

        # Add the positional embeddings to the embeddings
        embeddings += pos_embeddings

        # Process the new input through the decoder sequence and the final linear layer
        # with softmax activation and return the result
        return self._model(embeddings)

    def train_step(self, data_batch):
        # Get the input and the labels from the data batch
        X, y_true = data_batch

        # Perform a forward pass through the model, calculate the loss and then the gradients
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = self.loss(y_true, y_pred)

            gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the gradients to the model
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute the metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # Get the loss and the metrics
        metrics = {m.name: m.result() for m in self.metrics}

        return {"loss": loss, **metrics}

    def test_step(self, data_batch):
        X, y_true = data_batch

        y_pred = self(X, training=False)
        loss = self.loss(y_true, y_pred)

        # Compute the metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # Get the loss and the metrics
        metrics = {m.name: m.result() for m in self.metrics}

        return {"loss": loss, **metrics}

    def generate_tokens_from_input(self, sequence_length: int, initial_context: list[int] = None) -> list[int]:
        """
        Generate a sequence of tokens from a given input sequence
        :param initial_context: The input sequence to start the generation from
        :param sequence_length: The length for the desired output sequence
        :return: A list of integers representing the generated encoded sequence
        """

        if initial_context is None:
            initial_context = []

        # Get the maximum context length from the positional embedding layer
        max_context_length: int = self._pos_embedding_layer.input_dim

        # Generate the next token for the input sequence and
        # append it to the input sequence itself, using an autoregressive approach
        output: list[int] = []
        for _ in range(sequence_length):
            # Add the generated output to the initial_context
            input_batch: tf.Tensor = tf.concat([initial_context, output], axis=0)

            # If the input batch is longer than the maximum context length,
            # limit the input batch to the last max_context_length tokens,
            # since the positional embeddings are limited to this length
            if len(input_batch) > max_context_length:
                input_batch = input_batch[-max_context_length:]

            # Reshape the 1D input_batch tensor, into a 2D one (that is a batch with a single sequence).
            # The batch dimension is added to the input tensor, since the model expects a batch of sequences
            input_batch = tf.expand_dims(input_batch, axis=0)

            # Perform a forward pass through the model and
            # get a batch of predictions for each token in input_batch
            prediction_batch = self(input_batch, training=False)

            # Get a batch with the probabilities for the last token's prediction only
            last_token_probabilities_batch = prediction_batch[:, -1]

            # Convert the last token's probabilities to logits and sample the next token,
            # in order to introduce some randomness in the generation process and get a more natural text
            logits = tf.math.log(last_token_probabilities_batch)
            next_token: tf.Tensor = tf.random.categorical(logits, num_samples=1)

            # Remove the batch dimension from the next token,
            # since it's a batch of single token and not a batch of multiple tokens
            next_token = tf.squeeze(next_token)

            # Append the next token to the output sequence
            output.append(next_token.numpy())

        return output

    def summary(
            self,
            line_length=None,
            positions=None,
            print_fn=None,
            expand_nested=False,
            show_trainable=False,
            layer_range=None,
    ):
        self._model.summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range)

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        if save_format is None:
            save_format = os.path.splitext(filepath)[1].lstrip('.')

        super().save(filepath, overwrite, save_format, **kwargs)

    @staticmethod
    def load_model(filepath: str) -> 'Transformer':
        """
        Load a model from a given filepath
        :param filepath:
        :return:
        """

        # Load the model from the given filepath, specifying all the custom layers used in the model architecture
        return tf.keras.models.load_model(
            filepath, custom_objects={'Decoder': Decoder,
                                      'MultiHeadAttention': MultiHeadAttention,
                                      'HeadAttention': HeadAttention}
        )
