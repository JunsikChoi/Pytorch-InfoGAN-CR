import os
import torch
import numpy as np
import time
import datetime
import itertools
import torchvision.utils as vutils
from utils import *
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, config, data_loader):
        self.config = config
        self.dataset = config.dataset
        self.n_c_disc = config.n_c_disc
        self.dim_c_disc = config.dim_c_disc
        self.dim_c_cont = config.dim_c_cont
        self.dim_z = config.dim_z
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr_G = config.lr_G
        self.lr_D = config.lr_D
        self.lr_CR = config.lr_CR
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gpu_id = config.gpu_id
        self.num_epoch = config.num_epoch
        self.lambda_disc = config.lambda_disc
        self.lambda_cont = config.lambda_cont
        self.alpha = config.alpha
        self.log_step = config.log_step
        self.project_root = config.project_root
        self.model_name = config.model_name
        self.use_visdom = config.use_visdom

        self.data_loader = data_loader
        self.img_list = {}
        self._set_device(self.gpu_id)
        self.build_models()
        if self.use_visdom:
            self._set_plotter(config)
            self._set_logger()

    def _set_device(self, gpu_id):
        self.device = torch.device(gpu_id)

    def _set_plotter(self, config):
        self.plotter = VisdomPlotter(config)

    def _set_logger(self):
        self.logger = Logger()
        self.logger.create_target('step', 's', 'iterations')
        self.logger.create_target('Loss_G', 'G', 'Generator Loss')
        self.logger.create_target('Loss_D', 'D', 'Discriminator Loss')
        self.logger.create_target('Loss_Info', 'I', 'Info(G+L_d+L_c) Loss')
        self.logger.create_target('Loss_Disc', 'I_d', 'Discrete Code Loss')
        self.logger.create_target('Loss_CR', 'CR', 'Contrastive Loss')
        self.logger.create_target(
            'Prob_D', 'P_d_real', 'Prob of D for real / fake sample')
        self.logger.create_target(
            'Prob_D', 'P_d_fake', 'Prob of D for real / fake sample')
        self.logger.create_target(
            'Loss_Cont', 'I_c_total', 'Continuous Code Loss')
        for i in range(self.dim_c_cont):
            self.logger.create_target(
                'Loss_Cont', f'I_c_{i+1}', 'Continuous Code Loss')

        return

    def _sample(self):
        # Sample Z from N(0,1)
        z = torch.randn(self.batch_size, self.dim_z, device=self.device)

        # Sample discrete latent code from Cat(K=dim_c_disc)
        idx = np.zeros((self.n_c_disc, self.batch_size))
        c_disc = torch.zeros(self.batch_size, self.n_c_disc,
                             self.dim_c_disc, device=self.device)
        for i in range(self.n_c_disc):
            idx[i] = np.random.randint(self.dim_c_disc, size=self.batch_size)
            c_disc[torch.arange(0, self.batch_size), i, idx[i]] = 1.0

        # Sample continuous latent code from Unif(-1,1)
        c_cond = torch.rand(self.batch_size, self.dim_c_cont,
                            device=self.device) * 2 - 1

        # Concat z, c_disc, c_cond
        for i in range(self.n_c_disc):
            z = torch.cat((z, c_disc[:, i, :].squeeze()), dim=1)
        z = torch.cat((z, c_cond), dim=1)

        return z, idx

    def _sample_fixed_noise(self):
        # Sample Z from N(0,1)
        fixed_z = torch.randn(self.dim_c_disc*10, self.dim_z)

        # For each discrete variable, fix other discrete variable to 0 and random sample other variables.
        idx = np.arange(self.dim_c_disc).repeat(10)

        c_disc_list = []
        for i in range(self.n_c_disc):
            zero_template = torch.zeros(
                self.dim_c_disc*10, self.n_c_disc, self.dim_c_disc)
            # Set all the other discrete variable to Zero([1,0,...,0])
            for ii in range(self.n_c_disc):
                if (i == ii):
                    pass
                else:
                    zero_template[:, ii, 0] = 1.0
            for j in range(len(idx)):
                zero_template[np.arange(self.dim_c_disc*10), i, idx] = 1.0
            c_disc_list.append(zero_template)

        # Random sample continuous variables
        c_rand = torch.rand(self.dim_c_disc * 10, self.dim_c_cont) * 2 - 1
        c_range = torch.linspace(start=-1, end=1, steps=10)

        c_range_list = []
        for i in range(self.dim_c_disc):
            c_range_list.append(c_range)
        c_range = torch.cat(c_range_list, dim=0)

        c_cont_list = []
        for i in range(self.dim_c_cont):
            c_zero = torch.zeros(self.dim_c_disc * 10, self.dim_c_cont)
            c_zero[:, i] = c_range
            c_cont_list.append(c_zero)

        fixed_z_dict = {}
        for idx_c_disc in range(len(c_disc_list)):
            for idx_c_cont in range(len(c_cont_list)):
                z = fixed_z.clone()
                for j in range(self.n_c_disc):
                    z = torch.cat(
                        (z, c_disc_list[idx_c_disc][:, j, :].squeeze()), dim=1)
                z = torch.cat((z, c_cont_list[idx_c_cont]), dim=1)
                fixed_z_dict[(idx_c_disc, idx_c_cont)] = z
        return fixed_z_dict

    def build_models(self):
        if self.dataset == 'mnist':
            from models.mnist.discriminator import Discriminator
            from models.mnist.generator import Generator
            from models.mnist.cr_discriminator import CRDiscriminator
        elif self.dataset == 'dsprites':
            from models.dsprites.vanila.discriminator import Discriminator
            from models.dsprites.vanila.generator import Generator
            from models.dsprites.vanila.cr_discriminator import CRDiscriminator
        else:
            rasie(NotImplementedError)

        # Initiate Models
        self.G = Generator(self.dim_z, self.n_c_disc, self.dim_c_disc,
                           self.dim_c_cont).to(self.device)
        self.D = Discriminator(self.n_c_disc, self.dim_c_disc,
                               self.dim_c_cont).to(self.device)
        self.CR = CRDiscriminator(self.dim_c_cont).to(self.device)
        # Initialize
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        self.CR.apply(weights_init_normal)
        return

    def set_optimizer(self, param_list, lr):
        params_to_optimize = itertools.chain(*param_list)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params_to_optimize, lr=lr, betas=(self.beta1, self.beta2))
            return optimizer
        else:
            raise NotImplementedError

    def save_model(self, epoch):
        save_dir = os.path.join(
            self.project_root, f'results/{self.model_name}/checkpoint')
        os.makedirs(save_dir, exist_ok=True)

        # Save network weights.
        torch.save({
            'Generator': self.G.state_dict(),
            'Discriminator': self.D.state_dict(),
            'CRDiscriminator': self.CR.state_dict(),
            'configuations': self.config
        }, f'{save_dir}/Epoch_{epoch}.pth')

        return

    def _get_idx_fixed_z(self):
        idx_fixed = torch.from_numpy(np.array([np.random.randint(0, self.dim_c_cont)
                                               for i in range(self.batch_size)])).to(self.device)
        code_fixed = np.array([np.random.rand()
                               for i in range(self.batch_size)])
        latent_pair_1, _ = self._sample()
        latent_pair_2, _ = self._sample()
        for i in range(self.batch_size):
            latent_pair_1[i][-self.dim_c_cont+idx_fixed[i]] = code_fixed[i]
            latent_pair_2[i][-self.dim_c_cont+idx_fixed[i]] = code_fixed[i]
        z = torch.cat((latent_pair_1, latent_pair_2), dim=0)

        return z, idx_fixed

    def train(self):
        # Set opitmizers
        optim_G = self.set_optimizer([self.G.parameters(), self.D.module_Q.parameters(
        ), self.D.latent_disc.parameters(), self.D.latent_cont_mu.parameters()], lr=self.lr_G)
        optim_D = self.set_optimizer(
            [self.D.module_shared.parameters(), self.D.module_D.parameters()], lr=self.lr_D)
        optim_CR = self.set_optimizer(
            [self.CR.parameters(), self.G.parameters()], lr=self.lr_CR)

        # Loss functions
        adversarial_loss = torch.nn.BCELoss()
        categorical_loss = torch.nn.CrossEntropyLoss()
        continuous_loss = NLL_gaussian()

        # Sample fixed latent codes for comparison
        fixed_z_dict = self._sample_fixed_noise()

        start_time = time.time()
        num_steps = len(self.data_loader)
        step = 0
        log_target = {}
        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()
            step_epoch = 0
            for i, (data, _) in enumerate(self.data_loader, 0):
                if (data.size()[0] != self.batch_size):
                    self.batch_size = data.size()[0]

                data_real = data.to(self.device)

                # Update Discriminator
                # Reset optimizer
                optim_D.zero_grad()

                # Calculate Loss D(real)
                prob_real, _, _, _ = self.D(data_real)
                label_real = torch.full(
                    (self.batch_size,), 1, device=self.device)
                loss_D_real = adversarial_loss(prob_real, label_real)

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_real.backward()

                # calculate Loss D(fake)
                # Sample noise, latent codes
                z, idx = self._sample()
                data_fake = self.G(z)
                prob_fake_D, _, _, _ = self.D(data_fake.detach())
                label_fake = torch.full(
                    (self.batch_size,), 0, device=self.device)
                loss_D_fake = adversarial_loss(prob_fake_D, label_fake)

                # Calculate gradient -> grad accums to module_shared / modue_D
                loss_D_fake.backward()
                loss_D = loss_D_real + loss_D_fake

                # Update Parameters for D
                optim_D.step()

                # Update Generator and Q
                # Reset Optimizer
                optim_G.zero_grad()

                # Calculate loss for generator
                prob_fake, disc_logits, mu, var = self.D(data_fake)
                loss_G = adversarial_loss(prob_fake, label_real)

                # Calculate loss for discrete latent code
                target = torch.LongTensor(idx).to(self.device)
                loss_c_disc = 0
                for j in range(self.n_c_disc):
                    loss_c_disc += categorical_loss(
                        disc_logits[:, j, :], target[j, :])
                loss_c_disc = loss_c_disc * self.lambda_disc

                # Calculate loss for continuous latent code
                loss_c_cont = continuous_loss(
                    z[:, self.dim_z+self.n_c_disc*self.dim_c_disc:], mu, var).mean(0)

                loss_c_cont = loss_c_cont * self.lambda_cont

                loss_info = loss_G + loss_c_disc + loss_c_cont.sum()
                loss_info.backward()
                optim_G.step()

                optim_G.zero_grad()
                # Calculate contrastive loss
                idx_fixed_z, fixed_idx = self._get_idx_fixed_z()
                idx_fixed_data = self.G(idx_fixed_z)
                cr_logits = self.CR(
                    idx_fixed_data[:self.batch_size], idx_fixed_data[self.batch_size:])
                loss_cr = self.alpha * categorical_loss(cr_logits, fixed_idx)
                loss_cr.backward()
                optim_CR.step()

                # write data to log target
                if self.use_visdom:
                    self.logger.write('s', step)
                    self.logger.write('G', loss_G.item())
                    self.logger.write('D', loss_D.item())
                    self.logger.write('I', loss_info.item())
                    self.logger.write('I_d', loss_c_disc.item())
                    self.logger.write('CR', loss_cr.item())
                    self.logger.write('P_d_real', prob_real.mean().item())
                    self.logger.write('P_d_fake', prob_fake_D.mean().item())
                    self.logger.write('I_c_total', loss_c_cont.sum().item())
                    for c in range(self.dim_c_cont):
                        self.logger.write(f'I_c_{c+1}', loss_c_cont[c].item())

                # Print log info
                if (step % self.log_step == 0):
                    if self.use_visdom:
                        self.logger.pour_to_plotter(self.plotter)
                        self.logger.clear_data()

                    print('==========')
                    print(f'Model Name: {self.model_name}')
                    print('Epoch [%d/%d], Step [%d/%d], Elapsed Time: %s \nLoss D : %.4f, Loss_CR: %.4f, Loss Info: %.4f\nLoss_Disc: %.4f Loss_Cont: %.4f Loss_Gen: %.4f'
                          % (epoch + 1, self.num_epoch, step_epoch, num_steps, datetime.timedelta(seconds=time.time()-start_time), loss_D, loss_cr.item(), loss_info.item(), loss_c_disc.item(), loss_c_cont.sum().item(), loss_G.item()))
                    for c in range(len(loss_c_cont)):
                        print('Loss of %dth continuous latent code: %.4f' %
                              (c+1, loss_c_cont[c].item()))
                    print(
                        f'Prob_real_D:{prob_real.mean()}, Prob_fake_D:{prob_fake_D.mean()}, Prob_fake_G:{prob_fake.mean()}')

                step += 1
                step_epoch += 1

            # Plot and Log generated images
            for key in fixed_z_dict.keys():
                fixed_z = fixed_z_dict[key].to(self.device)
                idx_c_disc = key[0]
                idx_c_cont = key[1]

                # Generate and plot images from fixed inputs
                imgs, title = plot_generated_data(
                    self.config, self.G, fixed_z, epoch, idx_c_disc, idx_c_cont)

                # collect images for animation
                self.img_list[(epoch, key[0], key[1])] = vutils.make_grid(
                    imgs, nrow=10, padding=2, normalize=True)

                # Log Image
                if self.use_visdom:
                    self.plotter.plot_image_grid(title, imgs, title)

            # Save checkpoint
            self.save_model(epoch+1)

        # Make and save animation
        make_animation(self.config, epoch, self.n_c_disc,
                       self.dim_c_cont, self.img_list)
        return

    def test(self):
        pass
