# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, D2, Dhigh, diffaugment='', augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.D2 = D2
        self.Dhigh = Dhigh
        self.diffaugment = diffaugment
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def create_mask(self, ths, ref_fft):

        #null mask
        
        mask = torch.ones((ref_fft.shape), dtype=torch.bool)
        _, _, h, w = ref_fft.shape
        b_h = np.floor((h*ths)/2.0).astype(int)
        b_w = np.floor((w*ths)/2.0).astype(int)

        mask[:, :, 0:b_h, 0:b_w]     = 0      # top left
        mask[:, :, 0:b_h, w-b_w:w]   = 0      # top right
        mask[:, :, h-b_h:h, 0:b_w]   = 0      # bottom left
        mask[:, :, h-b_h:h, w-b_w:w] = 0      # bottom right

        return mask


    def fre_remove_random(self, img, attn):
        im_fft = torch.fft.fft2(img) #im: input image b, 3, h, w
        # im_fft = torch.fft.fft(im.clone(), 2) #im: input image b, 3, h, w

        b, c, im_h, im_w = im_fft.shape

        #decomposition
        ths=list(np.arange(0., 1+1./64, 1./64))
        each_fft = []
        # for i in range(len(ths)-1):
        for i in range(0,64):
            t1 = ths[i]
            t2 = ths[i+1]

            mask1 = self.create_mask(t1, im_fft)
            mask2 = self.create_mask(t2, im_fft)
            mask = torch.logical_xor(mask1, mask2)
            band_fft = im_fft * mask.to(im_fft.device)
            band_fft = torch.unsqueeze(band_fft, 1)
            each_fft.append(band_fft)



        fft_all = torch.cat(each_fft, dim=1) #b, 64, 3, w, h, 2

        #recover
        fft_sum = torch.sum(fft_all * attn, 1)
        
        reverse_img = torch.fft.ifft2(fft_sum).real
        # reverse_img = reverse_img.squeeze(0)
        return reverse_img


    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):

        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_Dhigh(self, img, c,  attn, sync):

        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        img = self.fre_remove_random(img, attn)
        with misc.ddp_sync(self.Dhigh, sync):
            logits = self.Dhigh(img, c)
        return logits

    def run_D2(self, img, c, sync):
        # if self.diffaugment:
        #     img = DiffAugment(img, policy=self.diffaugment)
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D2, sync):
            logits = self.D2(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img,attn, real_c, gen_z, gen_c, sync, gain, damping):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Dhighmain','Dhighreg', 'Dhighboth',  'D2main','D2reg', 'D2both']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dhighmain = (phase in ['Dhighmain', 'Dhighboth'])
        do_D2main = (phase in ['D2main', 'D2both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_Dhighr1   = (phase in ['Dhighreg', 'Dhighboth']) and (self.r1_gamma != 0)
        do_D2r1   = (phase in ['D2reg', 'D2both']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                gen_logits_high = self.run_Dhigh(gen_img, gen_c, attn, sync=False)
                training_stats.report('Loss/scores/fakehigh', gen_logits_high)
                training_stats.report('Loss/signs/fakehigh', gen_logits_high.sign())
                loss_Ghighmain = torch.nn.functional.softplus(-gen_logits_high) # -log(sigmoid(gen_logits))

                gen_logits_2 = self.run_D2(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake2', gen_logits_2)
                training_stats.report('Loss/signs/fake2', gen_logits_2.sign())
                loss_G2main = torch.nn.functional.softplus(-gen_logits_2) # -log(sigmoid(gen_logits))

                loss_G = loss_Gmain + loss_Ghighmain + loss_G2main

                training_stats.report('Loss/G/loss', loss_G)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_G.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img.detach(), gen_c.detach(), sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        loss_Dhighgen = 0
        if do_Dhighmain:
            with torch.autograd.profiler.record_function('Dhighgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_Dhigh(gen_img, gen_c,attn,  sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fakehigh', gen_logits)
                training_stats.report('Loss/signs/fakehigh', gen_logits.sign())
                loss_Dhighgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dhighgen_backward'):
                loss_Dhighgen.mean().mul(gain).backward()

        loss_D2gen = 0
        if do_D2main:
            with torch.autograd.profiler.record_function('D2gen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D2(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake2', gen_logits)
                training_stats.report('Loss/signs/fake2', gen_logits.sign())
                loss_D2gen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                W5 = None
                W6 = None
                for (w5, w6) in zip(self.D.parameters(), self.D2.parameters()):
                    if W5 is None and W6 is None:
                        W5 = w5.view(-1)
                        W6 = w6.view(-1)
                    else:
                        W5 = torch.cat((W5, w5.view(-1)), 0)
                        W6 = torch.cat((W6, w6.view(-1)), 0)
        
                loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1) # +1 is for a positive loss
                loss_weight = loss_weight * 0.01 * damping

                loss_D2 = loss_D2gen + loss_weight

            with torch.autograd.profiler.record_function('D2gen_backward'):
                loss_D2.mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        
        if do_Dhighmain or do_Dhighr1:
            name = 'Dhighreal_Dr1' if do_Dhighmain and do_Dhighr1 else 'Dhighreal' if do_Dhighmain else 'Dhighr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dhighr1)
                real_logits = self.run_Dhigh(real_img_tmp, real_c, attn, sync=sync)
                training_stats.report('Loss/scores/realhigh', real_logits)
                training_stats.report('Loss/signs/realhigh', real_logits.sign())

                loss_Dhighreal = 0
                if do_Dhighmain:
                    loss_Dhighreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/Dhigh/loss', loss_Dhighgen + loss_Dhighreal)

                loss_Dhighr1 = 0
                if do_Dhighr1:
                    with torch.autograd.profiler.record_function('rhigh1_grads'), conv2d_gradfix.no_weight_gradients():
                        rhigh1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    rhigh1_penalty = rhigh1_grads.square().sum([1,2,3])
                    loss_Dhighr1 = rhigh1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/rhigh1_penalty', rhigh1_penalty)
                    training_stats.report('Loss/Dhigh/reg', loss_Dhighr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dhighreal + loss_Dhighr1).mean().mul(gain).backward()


        if do_D2main or do_D2r1:
            name = 'D2real_Dr1' if do_D2main and do_D2r1 else 'D2real' if do_D2main else 'D2r1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_D2r1)
                real_logits = self.run_D2(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real2', real_logits)
                training_stats.report('Loss/signs/real2', real_logits.sign())

                loss_D2real = 0
                if do_D2main:
                    loss_D2real = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D2/loss', loss_D2 + loss_D2real)

                loss_D2r1 = 0
                if do_D2r1:
                    with torch.autograd.profiler.record_function('r21_grads'), conv2d_gradfix.no_weight_gradients():
                        r21_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r21_penalty = r21_grads.square().sum([1,2,3])
                    loss_D2r1 = r21_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r21_penalty', r21_penalty)
                    training_stats.report('Loss/D2/reg', loss_D2r1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_D2real + loss_D2r1).mean().mul(gain).backward()
#----------------------------------------------------------------------------
