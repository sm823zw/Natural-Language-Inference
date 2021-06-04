import tensorflow as tf

class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()

    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)

    def call(self, x):
        
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x,self.W)+self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return a, output
        
        return a, tf.keras.backend.sum(output, axis=1)
