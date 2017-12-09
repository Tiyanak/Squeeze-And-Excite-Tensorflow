import layers

config = {}

config['datasets'] = {1 : 'mnist', 2 : 'cifar', 3 : 'imagenet'}

# change this to change dataset used in application
config['dataset_name'] = config['datasets'][1]

# put corresponding dataset in corresponding data_dir in project/datasets/dataset/data_dir
DATA_DIR = 'datasets\\' + config['dataset_name'] + '\\data_dir'
SAVE_DIR = 'datasets\\' + config['dataset_name'] + '\\save_dir'
FILTERS_SAVE_DIR = 'datasets\\' + config['dataset_name'] + '\\save_dir\\filters'
PLOT_TRAINING_SAVE_DIR = 'datasets\\' + config['dataset_name'] + '\\save_dir\\plot_training'

# this is used in datasets classes as static values
config['mnist_img_width'] = 28
config['mnist_img_height'] = 28
config['mnist_img_channel'] = 1
config['cifar_img_width'] = 32
config['cifar_img_height'] = 32
config['cifar_img_channel'] = 3
config['imagenet_img_width'] = 256
config['imagenet_img_height'] = 256
config['imagenet_img_channel'] = 3

# this is used in application as current active dataset image config - dont touch
config['img_width'] = config[config['dataset_name'] + '_img_width']
config['img_height'] = config[config['dataset_name'] + '_img_height']
config['img_channel'] = config[config['dataset_name'] + '_img_channel']

# all available activation functions in application
config['activation_functions'] = {'selu': layers.selu, 'relu': layers.relu}
# choose which activation to use
config['activation_fn'] = 'relu'

# hiperparameters - free to change
config['learning_rate'] = 1e-1
config['batch_size'] = 50
config['max_epochs'] = 8
config['output_shape'] = 10
config['pool_size'] = [2, 2]
config['strides'] = 2
config['num_class'] = 10
config['use_se'] = True

# layers config
config['fc_1_output'] = 256
config['fc_2_output'] = 128
