from __future__ import print_function
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import random
import os
import scipy.io as sio

# load files
import model
import util
import classifier
import classifier_entropy
from config import opt

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

if opt.gzsl_od:
    print('Performing OD-based GZSL experiments!')

elif opt.gzsl:
    print('Performing Simple GZSL experiments!')
else:
    print('Performing ZSL experiments!')


# Init modules: Encoder, Generator, Discriminator
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt, opt.attSize)

print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)

# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# input_bce_att = torch.FloatTensor(opt.batch_size, opt.attSize)
one = torch.tensor(1, dtype=torch.float)
# one = torch.FloatTensor([1])
mone = one * -1

# Cuda
if opt.cuda:
    netG.cuda()
    netD.cuda()
    netE.cuda()
    netDec.cuda()
    netF.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    # input_bce_att = input_bce_att.cuda()
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

#saved_generated_feats = np.empty((0, opt.resSize))
#saved_generated_labels = np.empty((0, 1))
dict_saved_generated_label_feat = {}

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
            netDec.zero_grad()
            recons_fake = netDec(fake)
            # R_cost = WeightedL1(recons_fake, input_attv, bce=opt.bce_att, gt_bce=Variable(input_bce_att))
            R_cost = WeightedL1(recons_fake, input_attv)
            # Add reconstruction loss
            errG += opt.recons_weight * R_cost
            errG.backward()
            optimizerE.step()
            optimizerG.step()
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

    # TODO: Saving generated visual features for all unseen classes per epoch
    # size: 8192 * number of visual features & unseen classes * epoch
    # Example(HMDB51): 8192 * 800 * 25 * 100
    # syn_feature: torch.Size([20000, 8192])
    # syn_label: torch.Size([20000])
    #saved_generated_labels = np.hstack((saved_generated_labels, syn_label.resize(-1, 1)))
    #saved_generated_feats = np.hstack((saved_generated_feats, syn_feature))
    saved_generated_label_feat = np.hstack((syn_label, syn_feature.resize(syn_label.size(0), 1)))

    dict_saved_generated_label_feat[epoch] = saved_generated_label_feat

# save generated unseen visual feat.
saving_data_papth = '/content/drive/MyDrive/colab_data/KG_GCN_GAN'
sio.savemat(saving_data_papth + '/Unseen_Visual_Feat_' + opt.dataset + '_' +
            opt.class_embedding + '_' +
            'split_' + opt.split + '.mat',
            dict_saved_generated_label_feat)

# TODO: Generalized zero-shot learning
# TODO: Read generated unseen visual features from saved file
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


result_root = '/content/drive/MyDrive/colab_data/KG_GCN_GAN'
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

