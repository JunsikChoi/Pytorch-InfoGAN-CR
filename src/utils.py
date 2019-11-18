import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
import json
from visdom import Visdom

# Custom functions for training


class NLL_gaussian:
    def __call__(self, x, mu, var):
        '''
        Compute negative log-likelihood for Gaussian distribution
        '''
        eps = 1e-6
        l = (x - mu) ** 2
        l /= (2 * var + eps)
        l += 0.5 * torch.log(2 * np.pi * var + eps)

        return l


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Logging and Plotting

class Logger(object):
    def __init__(self):
        self.log_target = {}

    def write(self, split_name, value):
        self.log_target[split_name]['value'].append(value)
        return

    def create_target(self, name, split_name, caption):
        self.log_target[split_name] = {'caption': caption,
                                       'name': name, 'value': []}
        return

    def clear_data(self):
        for target in self.log_target.keys():
            self.log_target[target]['value'] = []

    def pour_to_plotter(self, plotter):
        for target in self.log_target.keys():
            if target != 's':
                plotter.plot_line(var_name=self.log_target[target]['name'], split_name=target, title_name=self.log_target[
                    target]['caption'], x=self.log_target['s']['value'], y=self.log_target[target]['value'])
        return


class VisdomPlotter(object):
    """Plots to Visdom"""

    def __init__(self, config):
        self.viz = Visdom(server=config.visdom_server)
        self.config = config
        self.env = config.model_name
        # self.env = 'test'
        self.plots = {}
        self._show_configurations()

    def _show_configurations(self):
        config_dict = vars(self.config)
        config_text = "<strong>Model Configurations</strong> </br>"
        for key in config_dict.keys():
            config_text = config_text + f'{key}: {config_dict[key]} </br>'
        self.viz.text(
            text=config_text, env=self.env)

    def plot_line(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array(x), Y=np.array(y), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='iterations',
                ylabel=var_name
            ))
            return
        else:
            self.viz.line(X=np.array(x), Y=np.array(
                y), env=self.env, win=self.plots[var_name], name=split_name, update='append')
            return

    def plot_image(self, var_name, img, caption):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.image(
                img=img, env=self.env, opts=dict(caption=caption, title=caption))
            return
        else:
            self.viz.image(
                img=img, win=self.plots[var_name], env=self.env, opts=dict(caption=caption, title=caption))
            return

    def plot_image_grid(self, var_name, imgs, caption):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(
                imgs, 10, 1, env=self.env, opts=dict(caption=caption, title=caption, xlabel='Continuous Code', ylabel='Discrete Code'))
            return
        else:
            self.viz.images(
                imgs, 10, 1, win=self.plots[var_name], env=self.env, opts=dict(caption=caption, title=caption, xlabel='Continuous Code', ylabel='Discrete Code'))
            return


def make_animation(config, end_epoch, n_c_disc, dim_c_cont, img_list):

    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 16, }

    save_dir = os.path.join(
        config.project_root, f'results/{config.model_name}/gifs')
    os.makedirs(save_dir, exist_ok=True)

    for c_d in range(n_c_disc):
        for c_c in range(dim_c_cont):
            fig = plt.figure(figsize=(10, 10))
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'Continuous Code Index = {c_c+1}', fontsize=20)
            plt.ylabel(f'Discrete Code Index = {c_d+1}', fontsize=20)
            ims = []
            for epoch in range(end_epoch+1):
                im = plt.imshow(np.transpose(
                    img_list[(epoch, c_d, c_c)].numpy(), (1, 2, 0)), animated=True)
                title = plt.text(
                    150, -10, f'Generated Images at Epoch: {epoch+1}', horizontalalignment='center', fontdict=font)
                ims.append([im, title])
                del(title)
            anim = animation.ArtistAnimation(
                fig, ims, interval=1000, repeat_delay=1000, blit=True)
            anim.save(
                f'{save_dir}/{config.model_name}-Cd_{c_d+1}-Cc_{c_c+1}.gif', dpi=80, writer='imagemagick')
            plt.show()

    return


def plot_generated_data(config, generator, z, epoch, idx_c_d, idx_c_c):
    with torch.no_grad():
        gen_data = generator(z).detach().cpu()
    title = f'Fixed_{config.model_name}_E-{epoch+1}_Cd-{idx_c_d}_Cc-{idx_c_c}'
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'Continuous Code Index = {idx_c_c}', fontsize=20)
    plt.ylabel(f'Discrete Code Index = {idx_c_d}', fontsize=20)
    plt.imshow(np.transpose(vutils.make_grid(
        gen_data, nrow=10, padding=2, normalize=True), (1, 2, 0)))
    result_dir = os.path.join(
        config.project_root, 'results', config.model_name, 'images')
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, title+'.png'))
    plt.close('all')
    return gen_data, title


# For debugging

# Extract existing gradient dictionary from model m
def extract_grad_dict(m):
    param_dict = {}
    for name, param in m.named_parameters():
        if param.grad is None:
            param_dict[name] = None
        else:
            param_dict[name] = param.grad.clone()
    return param_dict

# Compare two gradient dictionary


def compare_grad(d1, d2):
    different_name = []
    for key in d1.keys():
        if d1[key] is None:
            if d2[key] is not None:
                different_name.append(key)
            else:
                pass
        else:
            if not torch.equal(d1[key], d2[key]):
                different_name.append(key)
    if len(different_name) == 0:
        print("Same grad!")
    return different_name

# MISC


def save_config(config):

    save_dir = os.path.join(
        config.project_root, f'results/{config.model_name}')
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/config.json', 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

    return
