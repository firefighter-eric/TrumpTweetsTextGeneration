import tensorflow as tf


def build_basic_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ],
        name='BaseLineModel')
    return model


def build_attention_model(vocab_size, embedding_dim, rnn_units, batch_size):
    input_layer = tf.keras.Input(batch_shape=(batch_size, None))

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    query_embeddings = token_embedding(input_layer)
    value_embeddings = token_embedding(input_layer)

    # rnn
    gru_layer = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    stateful=True,
                                    recurrent_initializer='glorot_uniform')
    query_seq_encoding = gru_layer(query_embeddings)
    value_seq_encoding = gru_layer(value_embeddings)

    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Concatenate query and document encodings to produce a DNN input_layer layer.
    concentrate_layer = tf.keras.layers.Concatenate()([query_seq_encoding, query_value_attention_seq])
    dense = tf.keras.layers.Dense(vocab_size)
    output_layer = dense(concentrate_layer)

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer], name='ModelWithAttention')
    return model
