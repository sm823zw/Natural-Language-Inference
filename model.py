import tensorflow as tf
from attention import Attention

def create_model(premise, hypothesis, embed_matrix, l2, EMBEDDING_DIM, MAX_SEQ_LEN):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], output_dim=EMBEDDING_DIM, weights=[embed_matrix], input_length=MAX_SEQ_LEN, trainable=False)
    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))
    BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, kernel_regularizer=lam, recurrent_regularizer=lam, return_sequences=True))

    
    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = BiLSTM(premise)
    hypothesis = BiLSTM(hypothesis)

    premise = tf.keras.layers.BatchNormalization()(premise)
    hypothesis = tf.keras.layers.BatchNormalization()(hypothesis)

    _, premise = Attention(return_sequences=False)(premise)
    _, hypothesis = Attention(return_sequences=False)(hypothesis)

    dot_product = tf.keras.layers.Multiply()([premise, hypothesis])
    difference = tf.keras.layers.Subtract()([premise, hypothesis])

    train_input = tf.keras.layers.concatenate([premise, hypothesis, dot_product, difference])
    train_input = tf.keras.layers.Dropout(0.1)(train_input)

    for i in range(3):
        train_input = tf.keras.layers.Dense(200, kernel_regularizer=lam)(train_input)
        train_input = tf.keras.layers.BatchNormalization()(train_input)
        train_input = tf.keras.layers.ReLU()(train_input)
        train_input = tf.keras.layers.Dropout(0.1)(train_input)


    prediction = tf.keras.layers.Dense(3, activation='softmax')(train_input)

    return prediction
