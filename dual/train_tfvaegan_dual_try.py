from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import random
import os

# load files
import model_dual
import util_dual
import classifier_dual
import classifier_entropy_dual
from config_dual import opt

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
data = util_dual.DATA_LOADER(opt)
print("Training samples: ", data.ntrain)
print("Dataset: ", opt.dataset)
print("Dual GAN with ", opt.combined_syn)

if opt.gzsl_od:
    print('Performing OD-based GZSL experiments!')

elif opt.gzsl:
    print('Performing Simple GZSL experiments!')
else:
    print('Performing ZSL experiments!')


# Init modules: Encoder, Generator, Discriminator
netE_image = model_dual.Encoder(opt, semantics_type='image')
netG_image = model_dual.Generator(opt, semantics_type='image')
netD_image = model_dual.Discriminator_D1(opt, semantics_type='image')
# Init model_duals: Feedback module, auxillary module
netF_image = model_dual.Feedback(opt)
netDec_image = model_dual.AttDec(opt, opt.attSize_image)

netE_text = model_dual.Encoder(opt, semantics_type='text')
netG_text = model_dual.Generator(opt, semantics_type='text')
netD_text = model_dual.Discriminator_D1(opt, semantics_type='text')
# Init model_duals: Feedback module, auxillary module
netF_text = model_dual.Feedback(opt)
netDec_text = model_dual.AttDec(opt, opt.attSize_text)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)

input_att_text = torch.FloatTensor(opt.batch_size, opt.attSize_text)
input_att_image = torch.FloatTensor(opt.batch_size, opt.attSize_image)

noise_text = torch.FloatTensor(opt.batch_size, opt.nz_text)
noise_image = torch.FloatTensor(opt.batch_size, opt.nz_image)

# input_bce_att = torch.FloatTensor(opt.batch_size, opt.attSize)
one = torch.tensor(1, dtype=torch.float)
# one = torch.FloatTensor([1])
mone = one * -1

# Cuda
if opt.cuda:
    netG_image.cuda()
    netD_image.cuda()
    netE_image.cuda()
    netDec_image.cuda()
    netF_image.cuda()
    noise_image, input_att_image = noise_image.cuda(), input_att_image.cuda()

    netG_text.cuda()
    netD_text.cuda()
    netE_text.cuda()
    netDec_text.cuda()
    netF_text.cuda()
    noise_text, input_att_text = noise_text.cuda(), input_att_text.cuda()

    # input_bce_att = input_bce_att.cuda()
    input_res = input_res.cuda()
    one = one.cuda()
    mone = mone.cuda()


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
    batch_feature, batch_att_text, batch_att_image = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att_text.copy_(batch_att_text)
    input_att_image.copy_(batch_att_image)


def generate_syn_feature(netG, classes, attribute, num, netF=None, netDec=None, attSize=None, nz=None):
    # unseen feature synthesis
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, attSize)
    syn_noise = torch.FloatTensor(num, nz)
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
optimizerD_image = optim.Adam(netD_image.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE_image = optim.Adam(netE_image.parameters(), lr=opt.lr)
optimizerG_image = optim.Adam(netG_image.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF_image = optim.Adam(netF_image.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec_image = optim.Adam(netDec_image.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

optimizerD_text = optim.Adam(netD_text.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE_text = optim.Adam(netE_text.parameters(), lr=opt.lr)
optimizerG_text = optim.Adam(netG_text.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF_text = optim.Adam(netF_text.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec_text = optim.Adam(netDec_text.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))


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


# TODO: Recording best_acc, best_acc_per_class, best_cm
if opt.gzsl_od:
    best_gzsl_od_acc = 0

elif opt.gzsl:
    best_gzsl_simple_acc = 0

else:
    best_zsl_acc = 0
    best_zsl_acc_per_class = []
    best_zsl_cm = []

# Training loop
for epoch in range(0, opt.nepoch):
    print("Start VAEGAN Training at epoch: ", epoch)
    # TODO: Traing GAN-Image
    # feedback training loop
    for loop in range(0, opt.feedback_loop):
        print("Training GAN-Image..")
        for i in range(0, data.ntrain, opt.batch_size):
            # TODO: Discriminator training
            # unfreeze image discrimator
            for p in netD_image.parameters():
                p.requires_grad = True
            # unfreeze image deocder
            for p in netDec_image.parameters():
                p.requires_grad = True

            # unfreeze text discrimator
            for p in netD_text.parameters():
                p.requires_grad = True
            # unfreeze text deocder
            for p in netDec_text.parameters():
                p.requires_grad = True

            # TODO: Train D1 and Decoder
            gp_sum_image = 0
            gp_sum_text = 0
            for iter_d in range(opt.critic_iter):
                # TODO: BUG -> init? or train dual in one loop?
                sample()
                netD_image.zero_grad()
                input_resv_image = Variable(input_res)
                input_attv_image = Variable(input_att_image)

                netD_text.zero_grad()
                input_resv_text = Variable(input_res)
                input_attv_text = Variable(input_att_text)

                # TODO: Training the auxillary module
                netDec_image.zero_grad()
                recons_image = netDec_image(input_resv_image)
                R_cost_image = opt.recons_weight * WeightedL1(recons_image, input_attv_image)
                R_cost_image.backward()
                optimizerDec_image.step()
                criticD_real_image = netD_image(input_resv_image, input_attv_image)
                criticD_real_image = opt.gammaD * criticD_real_image.mean()
                criticD_real_image.backward(mone)

                netDec_text.zero_grad()
                recons_text = netDec_text(input_resv_text)
                R_cost_text = opt.recons_weight * WeightedL1(recons_text, input_attv_text)
                R_cost_text.backward()
                optimizerDec_text.step()
                criticD_real_text = netD_text(input_resv_text, input_attv_text)
                criticD_real_text = opt.gammaD * criticD_real_text.mean()
                criticD_real_text.backward(mone)

                if opt.encoded_noise:
                    means_image, log_var_image = netE_image(input_resv_image, input_attv_image)
                    std_image = torch.exp(0.5 * log_var_image)
                    eps_image = torch.randn([opt.batch_size, opt.latent_size_image])
                    if opt.cuda:
                        eps_image = eps_image.cuda()
                    eps_image = Variable(eps_image)
                    latent_code_image = eps_image * std_image + means_image

                    means_text, log_var_text = netE_text(input_resv_text, input_attv_text)
                    std_text = torch.exp(0.5 * log_var_text)
                    eps_text = torch.randn([opt.batch_size, opt.latent_size_text])
                    if opt.cuda:
                        eps_text = eps_text.cuda()
                    eps_text = Variable(eps_text)
                    latent_code_text = eps_text * std_text + means_text

                else:
                    noise_image.normal_(0, 1)
                    latent_code_image = Variable(noise_image)

                    noise_text.normal_(0, 1)
                    latent_code_text = Variable(noise_text)

                # TODO: feedback loop
                if loop == 1:
                    fake_image = feedback_module(gen_out=latent_code_image, att=input_attv_image,
                                           netG=netG_image, netDec=netDec_image, netF=netF_image)
                    fake_text = feedback_module(gen_out=latent_code_text, att=input_attv_text,
                                                netG=netG_text, netDec=netDec_text, netF=netF_text)
                else:
                    fake_image = netG_image(latent_code_image, c=input_attv_image)
                    fake_text = netG_text(latent_code_text, c=input_attv_text)

                criticD_fake_image = netD_image(fake_image.detach(), input_attv_image)
                criticD_fake_image = opt.gammaD * criticD_fake_image.mean()
                criticD_fake_image.backward(one)

                criticD_fake_text = netD_text(fake_text.detach(), input_attv_text)
                criticD_fake_text = opt.gammaD * criticD_fake_text.mean()
                criticD_fake_text.backward(one)

                # gradient penalty
                gradient_penalty_image = opt.gammaD * calc_gradient_penalty(netD_image, input_res, fake_image.data, input_att_image)
                gp_sum_image += gradient_penalty_image.data
                gradient_penalty_image.backward()
                Wasserstein_D_image = criticD_real_image - criticD_fake_image
                # add Y here
                # And add vae reconstruction loss
                D_cost_image = criticD_fake_image - criticD_real_image + gradient_penalty_image
                optimizerD_image.step()

                gradient_penalty_text = opt.gammaD * calc_gradient_penalty(netD_text, input_res, fake_text.data, input_att_text)
                gp_sum_text += gradient_penalty_text.data
                gradient_penalty_text.backward()
                Wasserstein_D_text= criticD_real_text - criticD_fake_text
                # add Y here
                # And add vae reconstruction loss
                D_cost_text = criticD_fake_text- criticD_real_text + gradient_penalty_text
                optimizerD_text.step()

            # Adaptive lambda
            gp_sum_image /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum_image > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum_image < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            gp_sum_text /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
            if (gp_sum_text > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum_text < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            # TODO:netG training
            # Train netG and Decoder
            for p in netD_image.parameters():
                p.requires_grad = False

            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec_image.parameters():
                    p.requires_grad = False

            for p in netD_text.parameters():
                p.requires_grad = False

            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec_text.parameters():
                    p.requires_grad = False

            netE_image.zero_grad()
            netG_image.zero_grad()
            netF_image.zero_grad()
            input_resv_image = Variable(input_res)
            input_attv_image = Variable(input_att_image)

            netE_text.zero_grad()
            netG_text.zero_grad()
            netF_text.zero_grad()
            input_resv_text = Variable(input_res)
            input_attv_text = Variable(input_att_text)

            # This is outside the opt.encoded_noise condition because of the vae loss
            means_image, log_var_image = netE_image(input_resv_image, input_attv_image)
            std_image = torch.exp(0.5 * log_var_image)
            eps_image = torch.randn([opt.batch_size, opt.latent_size_image])
            if opt.cuda:
                eps_image = eps_image.cuda()
            eps_image = Variable(eps_image)
            latent_code_image = eps_image * std_image + means_image

            means_text, log_var_text = netE_text(input_resv_text, input_attv_text)
            std_text = torch.exp(0.5 * log_var_text)
            eps_text = torch.randn([opt.batch_size, opt.latent_size_text])
            if opt.cuda:
                eps_text = eps_text.cuda()
            eps_text = Variable(eps_text)
            latent_code_text = eps_text * std_text + means_text

            if loop == 1:
                recon_x_image = feedback_module(gen_out=latent_code_image, att=input_attv_image,
                                          netG=netG_image, netDec=netDec_image, netF=netF_image)

                recon_x_text = feedback_module(gen_out=latent_code_text, att=input_attv_text,
                                          netG=netG_text, netDec=netDec_text, netF=netF_text)
            else:
                recon_x_image = netG_image(latent_code_image, c=input_attv_image)
                recon_x_text = netG_text(latent_code_text, c=input_attv_text)

            vae_loss_seen_image = loss_fn(recon_x_image, input_resv_image, means_image, log_var_image)
            errG_image = vae_loss_seen_image
            vae_loss_seen_text = loss_fn(recon_x_text, input_resv_text, means_text, log_var_text)
            errG_text = vae_loss_seen_text

            if opt.encoded_noise:
                criticG_fake_image = netD_image(recon_x_image, input_attv_image).mean()
                fake_image = recon_x_image
                criticG_fake_text = netD_text(recon_x_text, input_attv_text).mean()
                fake_text = recon_x_text
            else:
                noise_image.normal_(0, 1)
                latent_code_noise_image = Variable(noise_image)
                noise_text.normal_(0, 1)
                latent_code_noise_text = Variable(noise_text)

                if loop == 1:
                    fake_image = feedback_module(gen_out=latent_code_noise_image, att=input_attv_image,
                                           netG=netG_image, netDec=netDec_image, netF=netF_image)

                    fake_text = feedback_module(gen_out=latent_code_noise_text, att=input_attv_text,
                                                netG=netG_text, netDec=netDec_text, netF=netF_text)
                else:
                    fake_image = netG_image(latent_code_noise_image, c=input_attv_image)
                    fake_text = netG_text(latent_code_noise_text, c=input_attv_text)
                criticG_fake_image = netD_image(fake_image, input_attv_image).mean()
                criticG_fake_text = netD_text(fake_text, input_attv_text).mean()

            G_cost_image = -criticG_fake_image
            # Add vae loss and generator loss
            errG_image += opt.gammaG * G_cost_image
            netDec_image.zero_grad()
            recons_fake_image = netDec_image(fake_image)
            # R_cost = WeightedL1(recons_fake, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
            R_cost_image = WeightedL1(recons_fake_image, input_attv_image)
            # Add reconstruction loss
            errG_image += opt.recons_weight * R_cost_image
            errG_image.backward()
            optimizerE_image.step()
            optimizerG_image.step()


            G_cost_text = -criticG_fake_text
            # Add vae loss and generator loss
            errG_text += opt.gammaG * G_cost_text
            netDec_text.zero_grad()
            recons_fake_text = netDec_text(fake_text)
            # R_cost = WeightedL1(recons_fake, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
            R_cost_text = WeightedL1(recons_fake_text, input_attv_text)
            # Add reconstruction loss
            errG_text += opt.recons_weight * R_cost_text
            errG_text.backward()
            optimizerE_text.step()
            optimizerG_text.step()

            if loop == 1:
                optimizerF_image.step()
                optimizerF_text.step()
            # not train decoder at feedback time
            if opt.recons_weight > 0 and not opt.freeze_dec:
                optimizerDec_image.step()
                optimizerDec_text.step()
                # Print losses
    print('GAN-Image: [%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f \n'
          % (epoch, opt.nepoch, D_cost_image.data, G_cost_image.data, Wasserstein_D_image.data, vae_loss_seen_image.data), end=" ")
    # Evaluation
    netG_image.eval()
    netDec_image.eval()
    netF_image.eval()
    syn_feature_image, syn_label = generate_syn_feature(netG_image, data.unseenclasses, data.attribute_image,
                                                        opt.syn_num,
                                                        netF=netF_image, netDec=netDec_image,
                                                        attSize=opt.attSize_image, nz=opt.nz_image)
    print('GAN-Text: [%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f \n'
          % (epoch, opt.nepoch, D_cost_text.data, G_cost_text.data, Wasserstein_D_text.data, vae_loss_seen_text.data),
          end=" ")
    # Evaluation
    netG_text.eval()
    netDec_text.eval()
    netF_text.eval()
    syn_feature_text, _ = generate_syn_feature(netG_text, data.unseenclasses, data.attribute_text, opt.syn_num,
                                                       netF=netF_text, netDec=netDec_text,
                                                       attSize=opt.attSize_text, nz=opt.nz_text)

    if opt.combined_syn == 'concat':
        syn_feature = torch.cat((syn_feature_text, syn_feature_image), 1)
    else:
        print("Please choose the correct combination approaches.")

    # TODO: Generalized zero-shot learning
    if opt.gzsl_od:
        # OD based GZSL
        print("Performing Out-of-Distribution GZSL")
        seen_class = data.seenclasses.size(0)
        print('seen class size: ', seen_class)
        # TODO: not sure to use which netDec?
        clsu = classifier_dual.CLASSIFIER(syn_feature, util_dual.map_label(syn_label, data.unseenclasses),
                                     data, data.unseenclasses.size(0), opt.cuda,
                                     _nepoch=50, generalized=True,
                                     netDec=netDec_image, dec_size=opt.attSize_image, dec_hidden_size=4096)
        # _batch_size=opt.syn_num
        clss = classifier_dual.CLASSIFIER(data.train_feature, util_dual.map_label(data.train_label, data.seenclasses),
                                     data, data.seenclasses.size(0), opt.cuda,
                                     _nepoch=50, generalized=True,
                                     netDec=netDec_image, dec_size=opt.attSize_image, dec_hidden_size=4096)

        clsg = classifier_entropy_dual.CLASSIFIER(data.train_feature, util_dual.map_label(data.train_label, data.seenclasses),
                                             data, seen_class, syn_feature, syn_label,
                                             opt.cuda, clss, clsu, _batch_size=128,
                                             netDec=netDec_image, dec_size=opt.attSize_image, dec_hidden_size=4096)

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
        clsg = classifier_dual.CLASSIFIER(train_X, train_Y, data, nclass,
                                     opt.cuda, _nepoch=50,
                                     _batch_size=128, generalized=True,
                                     netDec=netDec_image, dec_size=opt.attSize_image, dec_hidden_size=4096)
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
        # Train ZSL classifier_dual
        zsl_cls = classifier_dual.CLASSIFIER(syn_feature, util_dual.map_label(syn_label, data.unseenclasses),
                                        data, data.unseenclasses.size(0),
                                        opt.cuda, opt.classifier_lr, 0.5, 50, opt.syn_num,
                                        generalized=False, netDec=netDec_image,
                                        dec_size=opt.attSize_image, dec_hidden_size=4096)
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
    netG_text.train()
    netDec_text.train()
    netF_text.train()

    netG_image.train()
    netDec_image.train()
    netF_image.train()

result_root = '/content/drive/MyDrive/colab_data/KG_GCN_GAN'
# Showing Best results
print('Showing Best Results for Dataset: ', opt.dataset)
# TODO: Save results into local file for ZSL, GZSL, GZSL-OD
if opt.gzsl_od:
    with open(os.path.join(result_root, "exp_gzsl_od_results.txt"), "a+") as f:
        f.write("\n" + "Dataset: " + str(opt.dataset) + "\n")
        f.write("Results: OD-based GZSL Experiments" + "\n")
        f.write("Split Index: " + str(opt.split) + "\n")

    print('Best GZSL-OD seen accuracy is', best_acc_seen, best_acc_per_seen)
    print('Best GZSL-OD unseen accuracy is', best_acc_unseen, best_acc_per_unseen)
    print('Best GZSL-OD H is', best_gzsl_od_acc)
    #print('Best GZSL-OD seen CM', best_cm_seen)
    #print('Best GZSL-OD unseen CM', best_cm_unseen)

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
                                        opt.class_embedding_text +
                                        opt.class_embedding_image + "_dual.txt"), "a+") as f:
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

