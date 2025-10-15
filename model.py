# model.py
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, ELU

def ProxyAUCLoss():
    """Pairwise ranking loss (AUC proxy)."""
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, "float32")
        y_true = tf.reshape(y_true, [-1])
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        pos = tf.boolean_mask(y_pred, tf.equal(y_true, 1))
        neg = tf.boolean_mask(y_pred, tf.equal(y_true, 0))

        def safe_loss():
            diffs = tf.expand_dims(neg, 0) - tf.expand_dims(pos, 1)
            return tf.reduce_mean(tf.nn.softplus(diffs))

        return tf.cond(
            tf.logical_and(tf.size(pos) > 0, tf.size(neg) > 0),
            safe_loss,
            lambda: 0.0,
        )
    return loss


def build_classifier_proxy(input_dim):
    """Rete neurale come in Schena et al. (2020)."""
    model = Sequential(name="IgAN_ProxyAUC_Classifier")
    for i in range(1, 5):
        model.add(Dense(100, input_dim=input_dim if i == 1 else None, name=f"Dense_{i}"))
        model.add(BatchNormalization(name=f"BN_{i}"))
        model.add(ELU(alpha=1.0, name=f"ELU_{i}"))
        model.add(Dropout(0.5, name=f"Dropout_{i}"))
    model.add(Dense(2, activation="softmax", name="Output"))
    return model
