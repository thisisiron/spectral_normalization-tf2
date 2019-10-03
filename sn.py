import tensorflow as tf  # TF 2.0


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, **kwargs):
        self.iteration = iteration
        self.eps = eps
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_variable(shape=(1, self.w_shape[-1]),
                                   initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                   trainable=False,
                                   name='sn_u',
                                   dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        return output
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        u_hat = self.u
        v_hat = None

        for i in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel.assign(self.w / sigma)
