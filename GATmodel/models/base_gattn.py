import tensorflow as tf

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        vars = tf.compat.v1.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)
        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)
    def masked_softmax_cross_entropy(logits, labels, mask):
        print("logits:",logits)
        print("labels:",labels)
        print("mask:",mask)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        predicted = tf.round(tf.nn.sigmoid(logits))
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)
        mask = tf.expand_dims(mask, -1)
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure
