from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
import random
from GCNmodel.utils import *
from GCNmodel.model import GCNModel
from GCNmodel.opt import Optimizer
from GATmodel.train import train
def LatentFeature(train_drug_microbe_matrix, drug_matrix, micorbe_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_microbe_matrix, drug_matrix, micorbe_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_microbe_matrix.sum()
    X = constructNet(train_drug_microbe_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    print(num_features)
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_microbe_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_microbe_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_microbe_matrix.shape[0], num_v=train_drug_microbe_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feature_representations=model.forward(sess,feed_dict)
    latent_features = feature_representations['final_embeddings']
    print(latent_features.shape)
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res,latent_features
if __name__ == "__main__":
    drug_sim = np.loadtxt('.\CODE\GCNATMDA\data\smiles gip.csv', delimiter=',',dtype="float32")
    microbe_sim = np.loadtxt('.\CODE\GCNATMDA\data\microbe_GIP_Sequence_sim.csv', delimiter=',',dtype="float32")
    drug_microbe_matrix = np.loadtxt('../GATmodel/data/MDA/interaction.csv', delimiter=',',dtype="int")
    epoch = 200
    emb_dim = 769
    lr = 0.0001
    adjdp = 0.7
    dp = 0.7
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    drug_microbe_res,finael_embeding = LatentFeature(
        drug_microbe_matrix, drug_sim*simw, microbe_sim*simw, random.seed(1), epoch, emb_dim, dp, lr, adjdp)
    kf = KFold(5, shuffle=True)
    tprs = []
    aucs = []
    accuracies = []
    prs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f1_scores,Recall,Precision = [],[],[]
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(627))):
        test_labels, scores = train(train_index, test_index,finael_embeding)

        fpr, tpr, _ = metrics.roc_curve(y_true=test_labels, y_score=scores)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        ax1.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
        precision, recall, _ = metrics.precision_recall_curve(y_true=test_labels, probas_pred=scores)
        Precision.append(precision)
        Recall.append(recall)
        prs.append(interp(mean_fpr, precision, recall))
        prs[-1][0] = 1.0
        pr_auc = metrics.auc(recall, precision)
        ax2.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUPR = %0.4f)' % (i, pr_auc))
        all_test_labels = np.concatenate(test_labels)
        all_scores = np.concatenate([np.ravel(score) for score in scores])
        accuracy = metrics.accuracy_score(y_true=all_test_labels, y_pred=(all_scores > 0.5).astype(int))
        accuracies.append(accuracy)
        f1 = metrics.f1_score(y_true=all_test_labels, y_pred=(all_scores > 0.5).astype(int))
        f1_scores.append(f1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    mean_accuracy = np.mean(accuracies)
    print(f'Average Accuracy = {mean_accuracy:.4f}')
    ax1.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    max_len = max(len(recall) for recall in Recall)
    adjusted_Recall = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(recall)), recall) for recall in
                       Recall]
    max_len = max(len(precision) for precision in Precision)
    adjusted_Precision = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(precision)), precision) for
                          precision in Precision]
    mean_precision = np.mean(adjusted_Precision)
    mean_recall = np.mean(adjusted_Recall)
    mean_f1 = np.mean(f1_scores)

    print(f'Average F1 Score = {mean_f1:.4f}')
    print(f'Average Recall = {mean_recall:.4f}')
    print(f'Average Precision = {mean_precision:.4f}')

    mean_precision = np.mean(prs, axis=0)
    mean_auc_pr = metrics.auc(mean_fpr, mean_precision)
    std_auc_pr = np.std(aucs)
    ax2.plot(mean_precision, mean_fpr, color='b',
             label=r'Mean PR (AUPR = %0.4f $\pm$ %0.2f)' % (mean_auc_pr, std_auc_pr),
             lw=2, alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver operating characteristic example')
    ax1.legend(loc="lower right")

    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall example')
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.show()







