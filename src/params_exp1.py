class Params:
    gpu = '0'
    in_dist = 'cifar10'

    # Contrastive loss with ODM
    num_feats = 10
    num_out = -1
    siam_margin = 10.0
    siam_batch_ratio = 0.25

    # Training params
    batch_size = 1024
    max_epochs = 3000
    lr_start = 1e-04
    lr_factor = 0.1
    lr_strat = [150, 2250]
    print_epoch = 1
    dropout_rate = 0.8
    w_decay = 1e-4

    # Files
    exp_name = 'EXP1_' + in_dist + '_v0'
    path_experiment = 'results/' + exp_name

    # Dataset Paths
    data_cifar10_path = 'data/cifar-10/cifar-10-batches-py'
    data_svhn_train_path = 'data/svhn/train_32x32.mat'
    data_svhn_test_path = 'data/svhn/test_32x32.mat'
    data_tiny_path = 'data/tiny_imagenet/'
    data_lsun_path = 'data/lsun/'
