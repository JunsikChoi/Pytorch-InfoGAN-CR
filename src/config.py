import argparse
import time
from pathlib import Path


def get_root():
    project_root = str(Path(__file__).resolve().parent.parent)
    return project_root


# def get_gpu_id():
#     return gpu_init(best_gpu_metric='mem', verbose=True)


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--dim_c_cont', type=int, default=5,
                     help="Dimension of continuous latent code")
net_arg.add_argument('--n_c_disc', type=int, default=0,
                     help="Number of categorical latent code")
net_arg.add_argument('--dim_c_disc', type=int, default=10,
                     help="Number of category in c_disc")
net_arg.add_argument('--dim_z', type=int, default=62,
                     help="Dimension of noise Z")
# net_arg.add_argument('--', type=, default=)

# Data
data_arg = add_argument_group('Data')
# data_arg.add_argument('--dataset', type=str, default='mnist')
data_arg.add_argument('--dataset', type=str, default='dsprites')
data_arg.add_argument('--batch_size', type=str, default=128)
data_arg.add_argument('--num_worker', type=int, default=12)
# data_arg.add_argument('--', type=, default=)

# Training
training_arg = add_argument_group('Training')
training_arg.add_argument('--is_train', type=str2bool, default=True)
training_arg.add_argument('--optimizer', type=str, default='adam')
training_arg.add_argument('--num_epoch', type=int, default=20)
training_arg.add_argument('--lr_G', type=float, default=0.001)
training_arg.add_argument('--lr_D', type=float, default=0.0002)
training_arg.add_argument('--lr_CR', type=float, default=0.001)
training_arg.add_argument('--beta1', type=float, default=0.5)
training_arg.add_argument('--beta2', type=float, default=0.999)
training_arg.add_argument('--lambda_disc', type=float, default=1)
training_arg.add_argument('--lambda_cont', type=float, default=0.1)
training_arg.add_argument('--alpha', type=float, default=0,
                          help="Hyperparameter for Contrastive Loss")

# Metric eval
metric_arg = add_argument_group('Metric')
metric_arg.add_argument('--num_eval_global_var', type=int, default=500)
metric_arg.add_argument('--eval_batch_size', type=int, default=100)


# Misc
misc_arg = add_argument_group('Misc')
# misc_arg.add_argument('--gpu_id', type=int, default=get_gpu_id())
misc_arg.add_argument('--log_step', type=int, default=10)
misc_arg.add_argument('--visdom_log_number', type=int, default=10,
                      help="How many times visdom logger send logging info to visdom server per epoch")
misc_arg.add_argument('--save_step', type=int, default=1,
                      help="Number of epochs for making checkpoint")
misc_arg.add_argument('--project_root', type=str, default=get_root())
misc_arg.add_argument('--model_name', type=str,
                      default='Debugging')
misc_arg.add_argument('--use_visdom', type=str2bool, default=True)
misc_arg.add_argument('--visdom_server', type=str,
                      default='http://192.168.192.99', help="Your visdom server address")


def get_config():
    config, unparsed = parser.parse_known_args()
    config.model_name += str(time.time())[-4:]
    return config, unparsed
