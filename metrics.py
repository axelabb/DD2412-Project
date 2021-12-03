
import tensorflow as tf

from tensorflow.keras.metrics import Mean,SparseCategoricalAccuracy
from tensorflow.keras.losses import Loss

class NLL(Loss):
    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        nll = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return nll



class NLL_metric(tf.keras.metrics.Metric):

    def __init__(self, n_classes,name="NLL", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.nll = self.add_weight(name='nll', initializer='zeros')

    def update_state(self, y_true,y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        nll = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        self.nll.assign_add(nll)

    def result(self):
        return self.nll

class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, n_classes,name="Accuracy", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.accuracy = self.add_weight(name='acc', initializer='zeros')
        self.n_classes = n_classes
        self.training = tf.Variable(True)

    def update_state(self, labels,logits):
        probs = tf.nn.softmax(tf.reshape(logits, [-1, self.n_classes]))
        if self.training:
            labels = tf.reshape(labels, [-1])
        else:
            probs = tf.math.reduce_mean(probs, axis=1) 

        accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels,probs)
        self.accuracy.assign_add(accuracy)

    def result(self):
        return self.accuracy


class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or 
     validation data is run during fit()) toggle metric flag '''
    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            if 'Accuracy' in metric.name:
                metric.on.assign(False)
    def on_test_end(self,  logs):
        for metric in self.model.metrics:
            if 'Accuracy' in metric.name:
                metric.on.assign(True)