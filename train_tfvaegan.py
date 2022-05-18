from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import random
import os
import pandas as pd

# load files
import model
import util
import classifier
import classifier_entropy
from config import opt
from center_loss import TripCenterLoss_min_margin, TripCenterLoss_margin
import time

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)
print("Dataset: ", opt.dataset)
print("Single GAN Experiments with FREE.")
print("Th number of generated unseen features per class: ", str(opt.syn_num))

if opt.gzsl_od:
    print('Performing OD-based GZSL experiments!')

elif opt.gzsl:
    print('Performing Simple GZSL experiments!')
else:
    print('Performing ZSL experiments!')


cls_criterion = nn.NLLLoss()
if opt.dataset in ['hmdb51']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen,
                                                   feat_dim=opt.attSize,
                                                   use_gpu=opt.cuda)

elif opt.dataset in ['ucf101']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen,
                                                   feat_dim=opt.attSize,
                                                   use_gpu=opt.cuda)
else:
    raise ValueError('Dataset %s is not supported'%(opt.dataset))

# Init modules: Encoder, Generator, Discriminator
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt, opt.attSize)
netFR = model.FR(opt, opt.attSize)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# input_bce_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
one = torch.tensor(1, dtype=torch.float)
# one = torch.FloatTensor([1])
mone = one * -1
beta=0

# Cuda: multi-GPU training
if opt.cuda:
    torch.nn.DataParallel(netG).cuda()
    torch.nn.DataParallel(netD).cuda()
    torch.nn.DataParallel(netE).cuda()
    torch.nn.DataParallel(netDec).cuda()
    torch.nn.DataParallel(netF).cuda()
    torch.nn.DataParallel(netFR).cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    # input_bce_att = input_bce_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label = input_label.cuda()


def loss_fn(recon_x, x, mean, log_var):
    # vae loss L_bce + L_kl
    BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), size_average=False)
    BCE = BCE.sum() / x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return BCE + KLD


def WeightedL1(pred, gt, bce=False, gt_bce=None):
    # semantic embedding cycle-consistency loss
    if bce:
        BCE = torch.nn.functional.binary_cross_entropy(pred + 1e-12, gt_bce.detach(), size_average=False)
        return BCE.sum() / pred.size(0)
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)


def feedback_module(gen_out, att, netG, netDec, netF):
    syn_fake = netG(gen_out, c=att)
    recons = netDec(syn_fake)
    recons_hidden_feat = netDec.getLayersOutDet()
    feedback_out = netF(recons_hidden_feat)
    syn_fake = netG(gen_out, a1=opt.a1, c=att, feedback_layers=feedback_out)
    return syn_fake


def sample():
    # data loader
    # batch_feature, batch_att, batch_bce_att = data.next_seen_batch(opt.batch_size)
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    # input_bce_att.copy_(batch_bce_att, batch_att)


def generate_syn_feature(netG, classes, attribute, num, netF=None, netDec=None):
    # unseen feature synthesis
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        # replicate the attributes
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)

        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)

        output = feedback_module(gen_out=syn_noisev, att=syn_attv, netG=netG, netDec=netDec, netF=netF)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizerFR = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizer_center = optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty_FR(netFR, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates,_ ,_, _ = netFR(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

    MI_loss = (torch.mean(kl_divergence) - i_c)

    return MI_loss


def optimize_beta(beta, MI_loss,alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))

    # return the updated beta value:
    return beta_new


# TODO: Recording best_acc, best_acc_per_class, best_cm
if opt.gzsl_od:
    best_gzsl_od_acc = 0
    print('Performing OD-based GZSL experiments!')

elif opt.gzsl:
    best_gzsl_simple_acc = 0
    print('Performing Simple GZSL experiments!')

elif opt.zsl:
    best_zsl_acc = 0
    best_zsl_acc_per_class = []
    best_zsl_cm = []
    print('Performing ZSL experiments!')
else:
    print('Wrong ZSL setting, please check if adding "zsl" or "gzsl" or "gzsl_od".')


# Training loop
for epoch in range(0, opt.nepoch):
    print("Start VAEGAN Training at epoch: ", epoch)
    # feedback training loop
    for loop in range(0, opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            # TODO: Discriminator training
            # unfreeze discrimator
            for p in netD.parameters():
                p.requires_grad = True

            # unfreeze deocder
            for p in netDec.parameters():
                p.requires_grad = True

            # TODO: Train D1 and Decoder
            gp_sum = 0
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                # TODO: Training the auxillary module
                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight * WeightedL1(recons, input_attv)
                # R_cost = opt.recons_weight*WeightedL1(recons, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD * criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size])
                    if opt.cuda:
                        eps = eps.cuda()
                    eps = Variable(eps)
                    latent_code = eps * std + means
                else:
                    noise.normal_(0, 1)
                    latent_code = Variable(noise)

                # TODO: feedback loop
                if loop == 1:
                    fake = feedback_module(gen_out=latent_code, att=input_attv, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake = netG(latent_code, c=input_attv)

                # TODO: update FR
                netFR.zero_grad()
                muR, varR, criticD_real_FR, latent_pred, _, recons_real = netFR(input_resv)
                # print("muR size: ", muR.size())
                # print("varR size: ", varR.size())
                # print("criticD_real_FR size: ", criticD_real_FR.size())
                # print("latent_pred size: ", latent_pred.size())
                # print("recons_real size: ", recons_real.size())
                criticD_real_FR = criticD_real_FR.mean()
                # recons_real should have the same size as input_resv_image (8192)
                R_cost = opt.recons_weight * WeightedL1(recons_real, input_attv)

                muF, varF, criticD_fake_FR, _, _, recons_fake = netFR(fake.detach())
                criticD_fake_FR = criticD_fake_FR.mean()
                gradient_penalty = calc_gradient_penalty_FR(netFR, input_resv, fake.data)
                center_loss_real = center_criterion(muR, input_label, margin=opt.center_margin,
                                                          incenter_weight=opt.incenter_weight)
                D_cost_FR = center_loss_real * opt.center_weight + R_cost
                D_cost_FR.backward()
                optimizerFR.step()
                optimizer_center.step()

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                # add Y here
                # And add vae reconstruction loss
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()

            # Adaptive lambda
            gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            # TODO:netG training
            # Train netG and Decoder
            for p in netD.parameters():
                p.requires_grad = False

            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters():
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            # This is outside the opt.encoded_noise condition because of the vae loss
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size])
            if opt.cuda:
                eps = eps.cuda()
            eps = Variable(eps)
            latent_code = eps * std + means
            if loop == 1:
                recon_x = feedback_module(gen_out=latent_code, att=input_attv, netG=netG, netDec=netDec, netF=netF)
            else:
                recon_x = netG(latent_code, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
            errG = vae_loss_seen

            if opt.encoded_noise:
                criticG_fake = netD(recon_x, input_attv).mean()
                fake = recon_x
            else:
                noise.normal_(0, 1)
                latent_code_noise = Variable(noise)
                if loop == 1:
                    fake = feedback_module(gen_out=latent_code_noise, att=input_attv, netG=netG, netDec=netDec,
                                           netF=netF)
                else:
                    fake = netG(latent_code_noise, c=input_attv)
                criticG_fake = netD(fake, input_attv).mean()

            G_cost = -criticG_fake
            # Add vae loss and generator loss
            errG += opt.gammaG * G_cost

            # netDec_image.zero_grad()
            # recons_fake = netDec_image(fake)
            netFR.zero_grad()
            _, _, criticG_fake_FR, latent_pred_fake, _, recons_fake = netFR(fake, train_G=True)
            # R_cost = WeightedL1(recons_fake, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
            R_cost = WeightedL1(recons_fake, input_attv)
            # Add reconstruction loss
            errG += opt.recons_weight * R_cost
            errG.backward()
            optimizerE.step()
            optimizerG.step()
            optimizerFR.step()
            if loop == 1:
                optimizerF.step()
            # not train decoder at feedback time
            if opt.recons_weight > 0 and not opt.freeze_dec:
                optimizerDec.step()
                # Print losses
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f \n'
          % (epoch, opt.nepoch, D_cost.data, G_cost.data, Wasserstein_D.data, vae_loss_seen.data), end=" ")
    # Evaluation
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num,
                                                  netF=netF, netDec=netDec)

    # TODO: Generalized zero-shot learning
    if opt.gzsl_od:
        # OD based GZSL
        print("Performing Out-of-Distribution GZSL")
        seen_class = data.seenclasses.size(0)
        clsu = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                     data, data.unseenclasses.size(0), opt.cuda,
                                     _nepoch=30, generalized=True, _batch_size=128,
                                     netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        # _batch_size=opt.syn_num
        clss = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data, data.seenclasses.size(0), opt.cuda,
                                     _nepoch=30, generalized=True, _batch_size=128,
                                     netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)

        clsg = classifier_entropy.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                             data, seen_class, syn_feature, syn_label,
                                             opt.cuda, clss, clsu, _nepoch=30, _batch_size=128,
                                             netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)

        if best_gzsl_od_acc < clsg.H:
            best_acc_seen, best_acc_unseen, best_gzsl_od_acc = clsg.acc_seen, clsg.acc_unseen, clsg.H
            best_acc_per_seen, best_acc_per_unseen = clsg.acc_per_seen, clsg.acc_per_unseen
            best_cm_seen, best_cm_unseen = clsg.cm_seen, clsg.cm_unseen
            best_epoch = epoch

        print('GZSL-OD: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f \n' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
        print('GZSL-OD: Acc per seen classes \n', clsg.acc_per_seen)
        print('GZSL-OD: Acc per unseen classes \n', clsg.acc_per_unseen)
        #print('GZSL-OD: seen confusion matrix: \n', clsg.cm_seen)
        #print('GZSL-OD: unseen confusion matrix: \n', clsg.cm_unseen)

    elif opt.gzsl:
        # TODO: simple Generalized zero-shot learning
        print("Performing simple GZSL")
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        clsg = classifier.CLASSIFIER(train_X, train_Y, data, nclass,
                                     opt.cuda, _nepoch=50,
                                     _batch_size=128, generalized=True,
                                     netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        if best_gzsl_simple_acc < clsg.H:
            best_acc_seen, best_acc_unseen, best_gzsl_simple_acc = clsg.acc_seen, clsg.acc_unseen, clsg.H
            best_acc_per_seen, best_acc_per_unseen = clsg.acc_per_seen, clsg.acc_per_unseen
            best_epoch = epoch
            # best_cm_seen, best_cm_unseen = clsg.cm_seen, clsg.cm_unseen

        print('Simple GZSL: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f \n' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
        print('Simple GZSL: Acc per seen classes \n', clsg.acc_per_seen)
        print('Simple GZSL: Acc per unseen classes \n', clsg.acc_per_unseen)
        #print('Simple GZSL: seen confusion matrix: \n', clsg.cm_seen)
        #print('Simple GZSL: unseen confusion matrix: \n', clsg.cm_unseen)

    else:
        # TODO: Zero-shot learning
        print("Performing ZSL")
        # Train ZSL classifier
        zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                        data, data.unseenclasses.size(0),
                                        opt.cuda, opt.classifier_lr, 0.5, 50, opt.syn_num,
                                        generalized=False, netDec=netDec,
                                        dec_size=opt.attSize, dec_hidden_size=4096)
        acc = zsl_cls.acc
        acc_per_class = zsl_cls.acc_per_class
        cm = zsl_cls.cm
        if best_zsl_acc < acc:
            best_zsl_acc = acc
            best_zsl_acc_per_class = acc_per_class
            best_zsl_cm = cm
            best_epoch = epoch
        print('ZSL unseen accuracy=%.4f at Epoch %d\n' % (acc, epoch))
        #print('ZSL unseen accuracy per class\n', acc_per_class)
        #print('ZSL confusion matrix\n', cm)

    # reset modules to training mode
    netG.train()
    netDec.train()
    netF.train()
    netFR.train()


result_root = opt.resultroot
# Showing Best results
print('Showing Best Results for Dataset: ', opt.dataset)
# TODO: Save results into local file for ZSL, GZSL, GZSL-OD
if opt.gzsl_od:
    with open(os.path.join(result_root, "exp_gzsl_od_results_" +
                                        opt.dataset + "_" +
                                        opt.class_embedding + ".txt"), "a+") as f:
        f.write("\n" + "Dataset: " + str(opt.dataset) + "\n")
        f.write("Results: OD-based GZSL Experiments" + "\n")
        f.write("Split Index: " + str(opt.split) + "\n")

        f.write("Visual Embedding: " + str(opt.action_embedding) + "\n")
        f.write("Semantic Embedding: " + str(opt.class_embedding) + "\n")

        # TODO: recording full confusion matrix
        f.write("Best Epoch: " + str(best_epoch) + "\n")
        f.write('Best GZSL-OD seen accuracy is ' + str(best_acc_seen) + "\n")
        f.write("Best GZSL-OD seen per-class accuracy: " + str(best_acc_per_seen) + "\n")

        f.write('Best GZSL-OD unseen accuracy is' + str(best_acc_unseen) + "\n")
        f.write("Best GZSL-OD unseen per-class accuracy: " + str(best_acc_per_unseen) + "\n")
        f.write('Best GZSL-OD H is ' + str(best_gzsl_od_acc) + "\n")

    print('Best GZSL-OD GZSL seen accuracy: ', best_acc_seen)
    print('Best GZSL-OD GZSL seen accuracy per class: ', best_acc_per_seen)

    print('Best GZSL-OD GZSL unseen accuracy: ', best_acc_unseen)
    print('Best GZSL-OD GZSL unseen accuracy per class: ', best_acc_per_unseen)
    print('Best GZSL-OD GZSL H: ', best_gzsl_od_acc)
    #print('Best GZSL-OD seen CM', best_cm_seen)
    #print('Best GZSL-OD unseen CM', best_cm_unseen)best_gzsl_od_acc

elif opt.gzsl:
    with open(os.path.join(result_root, "exp_gzsl_results.txt"), "a+") as f:
        f.write("\n" + "Dataset: " + str(opt.dataset) + "\n")
        f.write("Results: Simple GZSL Experiments" + "\n")
        f.write("Split Index: " + str(opt.split) + "\n")

        f.write("Visual Embedding: " + str(opt.action_embedding) + "\n")
        f.write("Semantic Embedding: " + str(opt.class_embedding) + "\n")

        # TODO: recording full confusion matrix
        f.write("Best Epoch: " + str(best_epoch) + "\n")
        f.write("Best Simple GZSL seen accuracy: " + str(best_acc_seen) + "\n")
        f.write("Best Simple GZSL seen per-class accuracy: " + str(best_acc_per_seen) + "\n")

        f.write("Best Simple GZSL unseen accuracy: " + str(best_acc_unseen) + "\n")
        f.write("Best Simple GZSL unseen per-class accuracy: " + str(best_acc_per_unseen) + "\n")
        f.write("Best Simple GZSL H: " + str(best_gzsl_simple_acc) + "\n")
        #f.write("Best ZSL unseen confusion matrix: " + str(best_zsl_cm) + "\n")

    print('Best Simple GZSL seen accuracy: ', best_acc_seen)
    print('Best Simple GZSL seen accuracy per class: ', best_acc_per_seen)

    print('Best Simple GZSL unseen accuracy: ', best_acc_unseen)
    print('Best Simple GZSL unseen accuracy per class: ', best_acc_per_unseen)
    print('Best Simple GZSL H: ', best_gzsl_simple_acc)
    #print('Best Simple GZSL seen CM', best_cm_seen)
    #print('Best Simple GZSL unseen CM', best_cm_unseen)

else:
    # ZSL:  best_zsl_acc
    #       best_zsl_acc_per_class,
    #       best_zsl_cm
    with open(os.path.join(result_root, "exp_zsl_results_" +
                                        opt.dataset + "_" +
                                        opt.class_embedding + ".txt"), "a+") as f:
        f.write("\n" + "Dataset: " + str(opt.dataset) + "\n")
        f.write("Results: ZSL Experiments" + "\n")
        f.write("Split Index: " + str(opt.split) + "\n")

        f.write("Visual Embedding: " + str(opt.action_embedding) + "\n")
        f.write("Semantic Embedding: " + str(opt.class_embedding) + "\n")

        # TODO: recording full confusion matrix
        f.write("Best Epoch: " + str(best_epoch) + "\n")
        f.write("Best ZSL unseen accuracy: " + str(best_zsl_acc) + "\n")
        f.write("Best ZSL unseen per-class accuracy: " + str(best_zsl_acc_per_class) + "\n")
        #f.write("Best ZSL unseen confusion matrix: " + str(best_zsl_cm) + "\n")

    print('Best ZSL unseen accuracy is', best_zsl_acc)
    print('Best ZSL unseen per-class accuracy is', best_zsl_acc_per_class)
    #print('Best ZSL unseen confusion matrix is', best_zsl_cm)

