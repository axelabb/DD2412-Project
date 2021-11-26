class NLL(Loss):
    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        nll = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return nll