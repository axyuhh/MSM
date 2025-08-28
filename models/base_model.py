import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import models.modules.network as network
import util.loss as loss
import util.util as util
from collections import OrderedDict
from torchvision.utils import make_grid
from abc import ABC, abstractmethod
from torch.autograd import Variable
import torch.nn.functional as F

################## BaseModel #############################
class BaseModel(ABC):
    def __init__(self, opt):
        if opt.gpu >= 0:
            self.device = torch.device('cuda:%d' % opt.gpu)
            torch.cuda.set_device(opt.gpu)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.opt = opt
        self.start_epoch = 0
        self.current_data = None
        self.nets = {}
        self.optimizers = {}
        self.lr_schedulers = {}
        self.build_models()

    def build_models(self):
        self.genAB = network.get_decoder()
        self.genBA = network.get_decoder()
        self.disA = network.get_discriminator()
        self.disB = network.get_discriminator()
        self.enc_styleA = network.get_style_encoder()
        self.enc_styleB = network.get_style_encoder()

        if self.opt.is_train:
            ckpt = None
            if self.opt.continue_train:
                ckpt = torch.load('{}/resume.pth'.format(self.opt.model_dir),map_location='cpu')
                print('sucess!!!!')
                self.start_epoch = ckpt['epoch']
            self._init_net(self.genAB, 'genAB', ckpt)
            self._init_net(self.genBA, 'genBA', ckpt)
            self._init_net(self.disA, 'disA', ckpt)
            self._init_net(self.disB, 'disB', ckpt)
            self._init_net(self.enc_styleA, 'enc_styleA', ckpt)
            self._init_net(self.enc_styleB, 'enc_styleB', ckpt)
            self.criterion_content = loss.get_gan_criterion(self.opt.content_gan_mode)
            self.criterion_image = loss.get_gan_criterion(self.opt.image_gan_mode)
            self.criterion_rec = loss.get_rec_loss(self.opt.rec_mode)
            self.criterion_kl = loss.get_kl_loss()
            self.cross_entropy_loss = nn.CrossEntropyLoss()

            self.print_networks(self.genAB)
            self.print_networks(self.disA)
        else:
            self._eval_net(self.genAB, 'genAB')
            self._eval_net(self.genBA, 'genBA')
            self._eval_net(self.enc_styleA, 'enc_styleA')
            self._eval_net(self.enc_styleB, 'enc_styleB')

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def _init_net(self, net, net_name, ckpt):
        net.to(self.device)
        net_optimizer = self.define_optimizer(net)
        if ckpt is not None:
            net.load_state_dict(ckpt[net_name]['weight'])
            net_optimizer.load_state_dict(ckpt[net_name]['optimizer'])
            lr_scheduler = util.get_scheduler(net_optimizer, self.opt, ckpt['epoch'])
        else:
            net.apply(network.weights_init(self.opt.init_type))
            lr_scheduler = util.get_scheduler(net_optimizer, self.opt, -1)
        net.train()
        self.nets[net_name] = net
        self.optimizers[net_name] = net_optimizer
        self.lr_schedulers[net_name] = lr_scheduler

    def _eval_net(self, net, net_name):
        net.load_state_dict(
            torch.load('{}/{}_{}.pth'.format(self.opt.model_dir, net_name, self.opt.which_epoch), map_location='cpu'))
        net.to(self.device)
        net.eval()

    def define_optimizer(self, net):
        return optim.Adam([{'params': net.parameters(), 'initial_lr': self.opt.lr}],
                          lr=self.opt.lr,
                          betas=(0.5, 0.999))

    def update_lr(self):
        for _, scheduler in self.lr_schedulers.items():
            scheduler.step()
        lr = self.optimizers['genAB'].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_generator(self, epoch):
        for net_name, net in self.nets.items():
            if 'dis' not in net_name:
                torch.save(net.state_dict(), '{}/{}_{}.pth'.format(self.opt.model_dir, net_name, epoch))

    def save_ckpt(self, epoch):
        ckpt = {'epoch': epoch}
        for net_name in self.nets:
            ckpt[net_name] = {'weight': self.nets[net_name].state_dict(),
                              'optimizer': self.optimizers[net_name].state_dict()}
        torch.save(ckpt, '{}/resume.pth'.format(self.opt.model_dir, net_name))

    @abstractmethod
    def prepare_data(self, data):
        '''prepare data for training or inference'''

    @abstractmethod
    def translation(self, data):
        '''translate the input image'''

    def get_current_errors(self):
        return self.current_losses

    def get_current_visuals(self):
        with torch.no_grad():
            self.genAB.eval()
            self.genBA.eval()
            self.enc_styleA.eval()
            self.enc_styleB.eval()
            real_A, real_B, _, _ = self.current_data

            styleA = self.enc_styleA(real_A)
            styleB = self.enc_styleB(real_B)

            fakeA = self.genBA(real_B, styleA)
            fakeB = self.genAB(real_A, styleB)

            recA = self.genBA(real_A, styleA)
            recB = self.genAB(real_B, styleB)

            fakeAB = self.genAB(fakeA, styleB)
            fakeBA = self.genBA(fakeB, styleA)

            self.genAB.train()
            self.genBA.train()
            self.enc_styleA.train()
            self.enc_styleB.train()
            imgs = torch.cat([torch.cat([real_A, fakeB, recA, fakeBA], dim=2), torch.cat([real_B, fakeA, recB, fakeAB], dim=2)], dim=3)
            imgs = make_grid(imgs, nrow=1)
            imgs = util.tensor2im(imgs)
            return {'real_A,fakeB,recA,fakeBA,real_B,fakeA,recB,fakeAB': imgs}

    def update_dis(self, dis, dis_opt, real, fake):
        dis.zero_grad()
        p_real_A = dis(real)
        p_real_B = dis(fake)
        errD = self.criterion_image(real=p_real_A, fake1=p_real_B) * 0.5
        errD.backward()
        dis_opt.step()
        return errD

    def calculate_gen_image(self, dis, fake):
        pred_fake = dis(fake)
        errG = self.criterion_image(real=pred_fake)
        return errG

    def update_model(self):
        ### prepare data ###
        real_A, real_B, source, target = self.current_data

        styleA = self.enc_styleA(real_A)
        styleB = self.enc_styleB(real_B)

        styleA_random = Variable(torch.randn(real_A.size(0), 8).cuda())
        styleB_random = Variable(torch.randn(real_A.size(0), 8).cuda())

        fakeA = self.genBA(real_B, styleA_random)
        fakeB = self.genAB(real_A, styleB_random)

        recA = self.genBA(real_A, styleA)
        recB = self.genAB(real_B, styleB)

        fakeAB = self.genAB(fakeA, styleB)
        fakeBA = self.genBA(fakeB, styleA)

        rec_styleA = self.enc_styleA(fakeA)
        rec_styleB = self.enc_styleB(fakeB)

        ### update discriminator ###
        errD_A = self.update_dis(self.disA,
                                    self.optimizers['disA'],
                                    real_A,
                                    fakeA.detach())
        errD_B = self.update_dis(self.disB,
                                    self.optimizers['disB'],
                                    real_B,
                                    fakeB.detach())

        ### update generator ###
        self.enc_styleA.zero_grad()
        self.enc_styleB.zero_grad()
        self.genBA.zero_grad()
        self.genAB.zero_grad()
        errGanA = self.calculate_gen_image(self.disA, fakeA)
        errGanB = self.calculate_gen_image(self.disB, fakeB)
        errStyleA1 = torch.mean(torch.abs(rec_styleA - styleA_random)) * self.opt.lambda_style
        errStyleB1 = torch.mean(torch.abs(rec_styleB - styleB_random)) * self.opt.lambda_style
        errCycB = self.criterion_rec(fakeAB, real_B) * 10
        errCycA = self.criterion_rec(fakeBA, real_A) * 10
        errIdtB = self.criterion_rec(recB, real_B) * 5
        errIdtA = self.criterion_rec(recA, real_A) * 5
        errG_total = errCycA + errCycB + errGanA + errGanB + errStyleA1 + errStyleB1 + errIdtB + errIdtA
        errG_total.backward()
        self.optimizers['genBA'].step()
        self.optimizers['genAB'].step()
        self.optimizers['enc_styleB'].step()
        self.optimizers['enc_styleA'].step()
        ###save current losses###
        dict = []
        dict += [('D_A', errD_A.item())]
        dict += [('G_A', errGanA.item())]
        dict += [('D_B', errD_B.item())]
        dict += [('G_B', errGanB.item())]
        dict += [('errStyleA1', errStyleA1.item())]
        dict += [('errStyleB1', errStyleB1.item())]
        dict += [('errCycA', errCycA.item())]
        dict += [('errCycB', errCycB.item())]
        dict += [('errIdtA', errIdtA.item())]
        dict += [('errIdtB', errIdtB.item())]
        self.current_losses = OrderedDict(dict)
