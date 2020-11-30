import numpy as np
import tensorflow as tf


def text2tf_data(filename, seq_length, batch_size, buffer_size):
    text = open(filename, 'rb').read().decode(encoding='utf-8')

    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    examples_per_epoch = len(text) // (seq_length + 1)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset, char2idx, idx2char, vocab
