image_size = 112
class_num = 2
batch_size = 32
device_param = "cuda:0"
learning_rate = 0.01
num_epochs = 300

"""Dir configuration"""
train_data_dir = '../data/train/'
val_data_dir = '../data/val/'
test_data_dir = '../data/test/'
log_path = '../log/'
model_path = '../model/'

"""Others"""
CRED = '\033[91m'
CEND = '\033[0m'