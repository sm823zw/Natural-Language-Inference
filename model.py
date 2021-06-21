import re
import tensorflow as tf
from attention import *

def create_LSTM_model(premise, hypothesis, 
                        embed_matrix, l2, 
                        EMBEDDING_DIM, MAX_SEQ_LEN, 
                        attention=False, baseline=True):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], 
                                            output_dim=EMBEDDING_DIM, 
                                            weights=[embed_matrix], 
                                            input_length=MAX_SEQ_LEN, 
                                            trainable=False)

    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))

    if attention:
        BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, 
                                                kernel_regularizer=lam, 
                                                recurrent_regularizer=lam, 
                                                return_sequences=True))
    else:
        BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, 
                                                kernel_regularizer=lam, 
                                                recurrent_regularizer=lam, 
                                                return_sequences=False))

    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = translation(premise)
    hypothesis = translation(hypothesis)

    premise = BiLSTM(premise)
    hypothesis = BiLSTM(hypothesis)

    if attention:
        _, premise = CustomAttention(return_sequences=False)(premise)
        _, hypothesis = CustomAttention(return_sequences=False)(hypothesis)

    if baseline:
        train_input = tf.keras.layers.concatenate([premise, hypothesis])
    else:
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


def create_GRU_model(premise, hypothesis, 
                        embed_matrix, l2, 
                        EMBEDDING_DIM, MAX_SEQ_LEN, 
                        attention=False, baseline=True):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], 
                                            output_dim=EMBEDDING_DIM, 
                                            weights=[embed_matrix], 
                                            input_length=MAX_SEQ_LEN, 
                                            trainable=False)

    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))
        

    if attention:
        BiGRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, 
                                                kernel_regularizer=lam, 
                                                recurrent_regularizer=lam, 
                                                return_sequences=True))
    else:
        BiGRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, 
                                                kernel_regularizer=lam, 
                                                recurrent_regularizer=lam, 
                                                return_sequences=False))
      
    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = BiGRU(premise)
    hypothesis = BiGRU(hypothesis)

    premise = tf.keras.layers.BatchNormalization()(premise)
    hypothesis = tf.keras.layers.BatchNormalization()(hypothesis)

    if attention:
        _, premise = CustomAttention(return_sequences=False)(premise)
        _, hypothesis = CustomAttention(return_sequences=False)(hypothesis)

    if baseline:
        train_input = tf.keras.layers.concatenate([premise, hypothesis])
    else:
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


def create_Rochtaschel_model(premise, hypothesis, 
                        embed_matrix, l2, 
                        EMBEDDING_DIM, MAX_SEQ_LEN,
                        two_way=False):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], 
                                            output_dim=EMBEDDING_DIM, 
                                            weights=[embed_matrix], 
                                            input_length=MAX_SEQ_LEN, 
                                            trainable=False)

    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))

    lstm_layer_1 = tf.keras.layers.LSTM(100, 
                                kernel_regularizer=lam, 
                                recurrent_regularizer=lam, 
                                return_sequences=True,
                                return_state=True,
                                time_major=False)

    lstm_layer_2 = tf.keras.layers.LSTM(100, 
                                kernel_regularizer=lam, 
                                recurrent_regularizer=lam, 
                                return_sequences=True,
                                time_major=False)

    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = translation(premise)
    hypothesis = translation(hypothesis)

    if two_way:
        premise_1, forward_h, forward_c, = lstm_layer_1(premise)
        init_states = [forward_h, forward_c]
        hypothesis_1 = lstm_layer_2(hypothesis, initial_state=init_states)
        train_input_1 = RochtaschelAttention(regularizer=lam)(tf.keras.layers.concatenate([premise_1, hypothesis_1], axis=1))
        hypothesis_2, forward_h, forward_c, = lstm_layer_1(hypothesis)
        init_states = [forward_h, forward_c]
        premise_2 = lstm_layer_2(premise, initial_state=init_states)
        train_input_2 = RochtaschelAttention(regularizer=lam)(tf.keras.layers.concatenate([hypothesis_2, premise_2], axis=1))
        train_input = tf.keras.layers.concatenate([train_input_1, train_input_2])
    else:
        premise, forward_h, forward_c, = lstm_layer_1(premise)
        init_states = [forward_h, forward_c]
        hypothesis = lstm_layer_2(hypothesis, initial_state=init_states)
        train_input = RochtaschelAttention(regularizer=lam)(tf.keras.layers.concatenate([premise, hypothesis], axis=1))


    train_input = tf.keras.layers.Dropout(0.25)(train_input)

    for i in range(3):
        train_input = tf.keras.layers.Dense(100, kernel_regularizer=lam)(train_input)
        train_input = tf.keras.layers.BatchNormalization()(train_input)
        train_input = tf.keras.layers.ReLU()(train_input)
        train_input = tf.keras.layers.Dropout(0.1)(train_input)


    prediction = tf.keras.layers.Dense(3, activation='softmax')(train_input)

    return prediction


def create_Inner_Attention_model(premise, hypothesis, 
                            embed_matrix, l2, 
                            EMBEDDING_DIM, MAX_SEQ_LEN, 
                            baseline=True):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], 
                                            output_dim=EMBEDDING_DIM, 
                                            weights=[embed_matrix], 
                                            input_length=MAX_SEQ_LEN, 
                                            trainable=False)

    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))

    BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, 
                                            kernel_regularizer=lam,
                                            recurrent_regularizer=lam, 
                                            return_sequences=True))

    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = translation(premise)
    hypothesis = translation(hypothesis)    

    premise = BiLSTM(premise)
    hypothesis = BiLSTM(hypothesis)

    premise = InnerAttention(regularizer=lam)(premise)
    hypothesis = InnerAttention(regularizer=lam)(hypothesis)

    if baseline:
        train_input = tf.keras.layers.concatenate([premise, hypothesis])
    else:
        dot_product = tf.keras.layers.Multiply()([premise, hypothesis])
        difference = tf.keras.layers.Subtract()([premise, hypothesis])
        train_input = tf.keras.layers.concatenate([premise, hypothesis, dot_product, difference])


    train_input = tf.keras.layers.Dropout(0.2)(train_input)

    for i in range(3):
        train_input = tf.keras.layers.Dense(100, kernel_regularizer=lam)(train_input)
        train_input = tf.keras.layers.BatchNormalization()(train_input)
        train_input = tf.keras.layers.ReLU()(train_input)
        train_input = tf.keras.layers.Dropout(0.2)(train_input)


    prediction = tf.keras.layers.Dense(3, activation='softmax')(train_input)

    return prediction

def create_Novel_model(premise, hypothesis, 
                            embed_matrix, l2, 
                            EMBEDDING_DIM, MAX_SEQ_LEN, 
                            baseline=True):

    lam = tf.keras.regularizers.l2(l2=l2)
    
    embedding =  tf.keras.layers.Embedding(embed_matrix.shape[0], 
                                            output_dim=EMBEDDING_DIM, 
                                            weights=[embed_matrix], 
                                            input_length=MAX_SEQ_LEN, 
                                            trainable=False)

    translation = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=lam))

    BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, 
                                            kernel_regularizer=lam,
                                            recurrent_regularizer=lam, 
                                            return_sequences=True))

    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = translation(premise)
    hypothesis = translation(hypothesis)

    premise = BiLSTM(premise)
    hypothesis = BiLSTM(hypothesis)

    premise_1 = InnerAttention(regularizer=lam)(premise)
    hypothesis_1 = InnerAttention(regularizer=lam)(hypothesis)

    _, premise_2 = CustomAttention(return_sequences=False, regularizer=lam)(premise)
    _, hypothesis_2 = CustomAttention(return_sequences=False, regularizer=lam)(hypothesis)

    if baseline:
        train_input = tf.keras.layers.concatenate([premise_1, hypothesis_1])
    else:
        dot_product = tf.keras.layers.Multiply()([premise_1, hypothesis_1])
        difference = tf.keras.layers.Subtract()([premise_1, hypothesis_1])
        train_input = tf.keras.layers.concatenate([premise_1, hypothesis_1, dot_product, difference, premise_2, hypothesis_2])


    train_input = tf.keras.layers.Dropout(0.2)(train_input)

    for i in range(3):
        train_input = tf.keras.layers.Dense(100, kernel_regularizer=lam)(train_input)
        train_input = tf.keras.layers.BatchNormalization()(train_input)
        train_input = tf.keras.layers.ReLU()(train_input)
        train_input = tf.keras.layers.Dropout(0.2)(train_input)


    prediction = tf.keras.layers.Dense(3, activation='softmax')(train_input)

    return prediction
