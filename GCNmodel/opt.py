from .clr import cyclic_learning_rate
import tensorflow as tf

class Optimizer():
    def __init__(self, model, preds, labels, lr, num_u, num_v, association_nam):
        preds_sub = preds
        labels_sub = labels

        global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.cost = tf.losses.mean_squared_error(
            labels=labels_sub, predictions=preds_sub)

        self.opt_op = self.optimizer.minimize(
            self.cost, global_step=global_step)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
