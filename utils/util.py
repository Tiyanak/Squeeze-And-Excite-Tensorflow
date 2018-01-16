import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as plt
import math
import skimage as ski
import skimage.io
from sklearn.metrics import confusion_matrix
from utils import constant
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def pickle_save(data, file):
    fo = open(file, 'wb')
    pickle.dump(data, fo)
    fo.close()

def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def class_to_onehot(Y, max_value):
    Yoh = np.zeros((len(Y), max_value))
    Yoh[range(len(Y)), Y] = 1
    return Yoh


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def plot_training_progress(data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['valid_loss'])
    x_data = np.linspace(1, num_points, num_points)

    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
             label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)

    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
             label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)

    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
             label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    print('Plotting in: ', constant.PLOT_FILE)
    plt.savefig(constant.PLOT_FILE)

    with open(constant.EVAL_RESULTS_FILE, 'w') as f:

        line = 'epoch_num,train_loss,valid_loss,train_acc,valid_acc,lr,epoch_time\n'
        f.write(line)

        for i in range(0, num_points):
            line = str(i+1) + data['train_loss'][i] + ',' + data['valid_loss'][i] + ',' + data['train_acc'][i] + ',' + \
                   data['valid_acc'][i] + ',' + data['lr'][i] + ',' + data['epoch_time'][i] + '\n'

            f.write(line)

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = confusion_matrix(Y, Y_)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((precision_i, recall_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))