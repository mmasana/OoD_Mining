import os
import pdb
import time
import math
import shutil
import numpy as np
import scipy.io as sio
import tensorflow as tf

# Local libraries
from params_exp1 import Params
import utils.utils_main as utma
import utils.utils_datasets as utdata
from models.vgg_net import Vggnet

# GPU options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = Params.gpu


def train_net(trn_x, trn_y, data_out):
    tf.reset_default_graph()
    # placeholders
    input = tf.placeholder(tf.float32, [None, trn_x.shape[1], trn_x.shape[2], trn_x.shape[3]])
    labels = tf.placeholder(tf.float32, [None])
    learning_rate = tf.placeholder(tf.float32, shape=[])
    dropout = tf.placeholder(tf.float32)
    # build network
    net = Vggnet(input, dropout, width_fc6=512, width_fc7=512, num_features=Params.num_feats)

    # add loss function -- for efficiency and not doubling the network's weights, we pass a batch of images and make the
    # pairs from it at the loss level. This next lines can be shortened, but are left as is for easier readability.
    left_p = tf.convert_to_tensor(range(0, Params.batch_size / 2), np.int32)
    rght_p = tf.convert_to_tensor(range(Params.batch_size / 2, Params.batch_size), np.int32)
    issame = tf.cast(tf.equal(tf.gather(labels, left_p), tf.gather(labels, rght_p)), tf.float32)
    dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.gather(net.out, left_p), tf.gather(net.out, rght_p))), 1))
    loss = tf.multiply((tf.constant(1.0) - issame), tf.nn.relu(Params.siam_margin - dist))
    loss = loss + tf.multiply(issame, dist)
    loss = tf.constant(0.5) * tf.reduce_mean(loss)
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = loss + l2 * Params.w_decay

    # choose optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables())

    # actual training
    with tf.Session(config=config) as sess:
        # inits
        loss_batch, aux_batch = [], []
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # load previous trained model if exists
        checkpoint_dir = Params.path_experiment
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(' * Checkpoint found for trained model')
        else:
            print(' * No checkpoint found for trained model')
        # epoch training loop
        lr = Params.lr_start
        for epoch in range(Params.max_epochs):
            epoch_time = time.time()

            # Split data into batches
            batch_count, batch_x, batch_y = utma.epoch_batches(trn_x, trn_y, data_out, Params)

            # batch training loop
            for m in range(batch_count):
                loss1, _, aux1 = sess.run([loss, train_op, issame], feed_dict={input: batch_x[m], labels: batch_y[m],
                                                                               learning_rate: lr,
                                                                               dropout: Params.dropout_rate})
                loss_batch.append(loss1)
                aux_batch.append(aux1)

            # save model -- with sanity check for when the loss breaks
            if math.isnan(np.mean(loss_batch)):
                break
            else:
                saver.save(sess, os.path.join(Params.path_experiment, '_model.ckpt'), global_step=epoch)

            # decrease learning rate by a factor every certain amount of epochs
            if epoch in Params.lr_strat:
                lr *= Params.lr_factor
            if epoch % Params.print_epoch == 0:
                print("Epoch {}: loss {} -- pairs {}:{} -- time {}".format(epoch,  np.mean(loss_batch),
                                                                           np.mean(np.sum(np.mean(aux_batch))),
                                                                           np.mean(np.sum(1-np.mean(aux_batch))),
                                                                           time.time() - epoch_time))
                loss_batch = []
                aux_batch = []


def extract_embeddings(data_x):
    tf.reset_default_graph()
    input = tf.placeholder(tf.float32, [None, data_x.shape[1], data_x.shape[2], data_x.shape[3]])
    dropout = tf.placeholder(tf.float32)
    net = Vggnet(input, dropout, width_fc6=512, width_fc7=512, num_features=Params.num_feats)
    # evaluate network accuracy
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        # Load trained model
        checkpoint_dir = Params.path_experiment
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Error: no checkpoint found for trained model')
        # Evaluate
        embeddings = np.zeros([data_x.shape[0], Params.num_feats])
        for m in range(data_x.shape[0]):
            embeddings[m, :] = net.out.eval({input: [data_x[m]], dropout: 1.0})
    return embeddings


def test_net(in_embed_trn, trn_y, in_embed_tst, tst_y):
    cluster_mean = utma.calculate_centers(in_embed_trn, trn_y)
    pred = np.zeros(in_embed_tst.shape[0]).astype(int)
    for m in range(in_embed_tst.shape[0]):
        pred[m] = np.argmin(np.sum(np.square(in_embed_tst[m] - cluster_mean), 1))
    acc_test = (100.0 * np.sum(pred == tst_y)) / float(in_embed_tst.shape[0])
    print('---')
    print ("Accuracy on test set is {}".format(acc_test))


def basic_baseline(in_embed_trn, trn_y, in_embed_tst, out_embed_tst):
    cluster_mean = utma.calculate_centers(in_embed_trn, trn_y)
    in_tst = np.empty(len(in_embed_tst))
    out_tst = np.empty(len(out_embed_tst))
    for m in range(in_embed_tst.shape[0]):
        in_tst[m] = np.min(np.sum(np.square(in_embed_tst[m] - cluster_mean), 1))
    for m in range(out_embed_tst.shape[0]):
        out_tst[m] = np.min(np.sum(np.square(out_embed_tst[m] - cluster_mean), 1))
    y_score = np.concatenate([in_tst, out_tst])
    y_score = 1.0 - y_score / np.max(y_score)
    y_true = np.array(len(in_tst) * [1] + len(out_tst) * [0]).astype(int)
    utma.calculate_metrics_B(y_score, y_true)


if __name__ == '__main__':
    # Check if experiment folder already exists
    if not os.path.exists(Params.path_experiment):
        os.makedirs(Params.path_experiment)
    # copy current params to experiment folder
    shutil.copy('params_exp1.py', Params.path_experiment)

    # Load CIFAR-10
    data_dir = Params.data_cifar10_path
    train_files = ['data_batch_%d' % d for d in xrange(1, 6)]
    c10_trn_x, c10_trn_y = utdata.load_cifar10_set(train_files, data_dir)
    pi = np.random.permutation(len(c10_trn_x))
    c10_trn_x, c10_trn_y = c10_trn_x[pi], c10_trn_y[pi]
    c10_tst_x, c10_tst_y = utdata.load_cifar10_set(['test_batch'], data_dir)
    c10_tst_y = np.asarray(c10_tst_y)

    # Load SVHN
    data = sio.loadmat(Params.data_svhn_train_path)
    svhn_trn_x = data['X']
    svhn_trn_y = data['y'] - 1
    svhn_trn_x = svhn_trn_x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1) / 255.0
    svhn_trn_y = np.squeeze(svhn_trn_y)
    data = sio.loadmat(Params.data_svhn_test_path)
    svhn_tst_x = data['X']
    svhn_tst_y = data['y'] - 1
    svhn_tst_x = svhn_tst_x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1) / 255.0
    svhn_tst_y = np.squeeze(svhn_tst_y)

    # Other datasets
    tiny_path = Params.data_tiny_path
    lsun_path = Params.data_lsun_path
    tiny_x = utdata.load_test(tiny_path, 32)
    lsun_x = utdata.load_test(lsun_path, 32)
    noise_x = np.random.normal(0.5, 1.0, (c10_trn_x.shape[0], c10_trn_x.shape[1], c10_trn_x.shape[2], c10_trn_x.shape[3]))
    noise_x = np.clip(noise_x, 0.0, 1.0)

    print('In: ' + Params.in_dist)
    if Params.in_dist == 'cifar10':
        print('Out: svhn')
        in_trn_x, in_trn_y = c10_trn_x, c10_trn_y
        in_tst_x, in_tst_y = c10_tst_x, c10_tst_y
        out_trn_x = svhn_trn_x
        out_tst_x = svhn_tst_x
    elif Params.in_dist == 'svhn':
        print('Out: cifar10')
        in_trn_x, in_trn_y = svhn_trn_x, svhn_trn_y
        in_tst_x, in_tst_y = svhn_tst_x, svhn_tst_y
        out_trn_x = c10_trn_x
        out_tst_x = c10_tst_x

    # Train
    train_net(in_trn_x, in_trn_y, out_trn_x)
    # Embeddings
    in_embed_trn = extract_embeddings(in_trn_x)
    in_embed_tst = extract_embeddings(in_tst_x)
    out_embed_tst = extract_embeddings(out_tst_x)
    tiny_embed_tst = extract_embeddings(tiny_x)
    lsun_embed_tst = extract_embeddings(lsun_x)
    ano1_embed_tst = extract_embeddings(noise_x)
    # Test
    test_net(in_embed_trn, in_trn_y, in_embed_tst, in_tst_y)
    # Novelty metrics
    basic_baseline(in_embed_trn, in_trn_y, in_embed_tst, out_embed_tst)
    # TinyImageNet
    basic_baseline(in_embed_trn, in_trn_y, in_embed_tst, tiny_embed_tst)
    # LSUN
    basic_baseline(in_embed_trn, in_trn_y, in_embed_tst, lsun_embed_tst)
    # Random noise
    basic_baseline(in_embed_trn, in_trn_y, in_embed_tst, ano1_embed_tst)

    print(Params.path_experiment)
