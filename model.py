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

    premise = BiLSTM(premise)
    hypothesis = BiLSTM(hypothesis)

    premise = tf.keras.layers.BatchNormalization()(premise)
    hypothesis = tf.keras.layers.BatchNormalization()(hypothesis)

    if attention:
        _, premise = Attention(return_sequences=False)(premise)
        _, hypothesis = Attention(return_sequences=False)(hypothesis)

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
        _, premise = Attention(return_sequences=False)(premise)
        _, hypothesis = Attention(return_sequences=False)(hypothesis)

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

    LSTM_p = tf.keras.layers.LSTM(100, 
                                kernel_regularizer=lam, 
                                recurrent_regularizer=lam, 
                                return_sequences=True,
                                return_state=True,
                                time_major=False)

    LSTM_h = tf.keras.layers.LSTM(100,
                                kernel_regularizer=lam, 
                                recurrent_regularizer=lam,
                                return_sequences=True,
                                time_major=False)

    premise = embedding(premise)
    hypothesis = embedding(hypothesis)

    premise = translation(premise)
    hypothesis = translation(hypothesis)

    if two_way:
        premise_1, forward_h, forward_c, = LSTM_p(premise)
        init_states = [forward_h, forward_c]
        hypothesis_1 = LSTM_h(hypothesis, initial_state=init_states)
        train_input_1 = RochtaschelAttention()(tf.keras.layers.concatenate([premise_1, hypothesis_1], axis=1))
        hypothesis_2, forward_h, forward_c, = LSTM_p(hypothesis)
        init_states = [forward_h, forward_c]
        premise_2 = LSTM_h(premise, initial_state=init_states)
        train_input_2 = RochtaschelAttention()(tf.keras.layers.concatenate([hypothesis_2, premise_2], axis=1))
        train_input = tf.keras.layers.concatenate([train_input_1, train_input_2])
    else:
        premise, forward_h, forward_c, = LSTM_p(premise)
        init_states = [forward_h, forward_c]
        hypothesis = LSTM_h(hypothesis, initial_state=init_states)
        train_input = RochtaschelAttention()(tf.keras.layers.concatenate([premise, hypothesis], axis=1))


    train_input = tf.keras.layers.Dropout(0.1)(train_input)

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

    premise = InnerAttention()(premise)
    hypothesis = InnerAttention()(hypothesis)

    if baseline:
        train_input = tf.keras.layers.concatenate([premise, hypothesis])
    else:
        dot_product = tf.keras.layers.Multiply()([premise, hypothesis])
        difference = tf.keras.layers.Subtract()([premise, hypothesis])
        train_input = tf.keras.layers.concatenate([premise, hypothesis, dot_product, difference])


    train_input = tf.keras.layers.Dropout(0.1)(train_input)

    for i in range(3):
        train_input = tf.keras.layers.Dense(100, kernel_regularizer=lam)(train_input)
        train_input = tf.keras.layers.BatchNormalization()(train_input)
        train_input = tf.keras.layers.ReLU()(train_input)
        train_input = tf.keras.layers.Dropout(0.1)(train_input)


    prediction = tf.keras.layers.Dense(3, activation='softmax')(train_input)

    return prediction
