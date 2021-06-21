import tensorflow as tf

class CustomAttention(tf.keras.layers.Layer):

    def __init__(self, regularizer, return_sequences=True, name=None, **kwargs):
        super(CustomAttention, self).__init__(name=name)
        self.kernel_regularizer = regularizer
        self.return_sequences = return_sequences
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        
        super(CustomAttention,self).build(input_shape)

    def call(self, x):
        
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        
        if self.return_sequences:
            return a, output
        
        return a, tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regu;arizer': self.kernel_regularizer,
            'return_sequences': self.return_sequences 
        })
        return config


class RochtaschelAttention(tf.keras.layers.Layer):

    def __init__(self, regularizer, name=None, **kwargs):
        super(RochtaschelAttention, self).__init__(name=name)
        self.kernel_regularizer = regularizer
        super(RochtaschelAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W_y = self.add_weight(name="att_weight_y", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.W_h = self.add_weight(name="att_weight_h", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.W_p = self.add_weight(name="att_weight_p", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.W_x = self.add_weight(name="att_weight_x", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.w = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)

        return super(RochtaschelAttention, self).build(input_shape)


    def call(self, inputs):

        y = inputs[:, :inputs.shape[1]//2, :]
        h_n = inputs[:, -1, :]
        first_term =  tf.einsum('ijk,kk->ijk', y, self.W_y)
        second_term = tf.keras.layers.RepeatVector(inputs.shape[1]//2)(tf.einsum('ik,kk->ik', h_n, self.W_h))
        M = tf.keras.activations.tanh(first_term + second_term)
        alpha = tf.keras.activations.softmax(tf.einsum('ij,lkj->lik', tf.transpose(self.w), M))
        r = tf.einsum('ijk,ilj->ik', y, alpha)
        output = tf.keras.activations.tanh(tf.einsum('jj,ij->ij', self.W_p, r) + tf.einsum('jj,ij->ij', self.W_x, h_n))
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regularizer': self.kernel_regularizer 
        })
        return config


class InnerAttention(tf.keras.layers.Layer):

    def __init__(self, regularizer, name=None, **kwargs):
        super(InnerAttention, self).__init__(name=name)
        self.kernel_regularizer = regularizer
        super(InnerAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W_y = self.add_weight(name="att_weight_y", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.W_h = self.add_weight(name="att_weight_h", shape=(input_shape[-1], input_shape[-1]),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)
        self.w = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="glorot_uniform", regularizer=self.kernel_regularizer, trainable=True)

        return super(InnerAttention, self).build(input_shape)

    def call(self, y):

        R_avg = tf.keras.layers.GlobalAveragePooling1D()(y)
        first_term = tf.einsum('ijk,kk->ijk', y, self.W_y)
        second_term = tf.keras.layers.RepeatVector(y.shape[1])(tf.einsum('ik,kk->ik', R_avg, self.W_h))
        M = tf.keras.activations.tanh(first_term + second_term)
        alpha = tf.keras.activations.softmax(tf.einsum('ij,lkj->lik', tf.transpose(self.w), M))
        output = tf.einsum('ijk,ilj->ik', y, alpha)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regularizer': self.kernel_regularizer 
        })
        return config
