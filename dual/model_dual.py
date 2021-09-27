import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Encoder
class Encoder(nn.Module):
    def __init__(self, opt, semantics_type=None):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        if semantics_type == 'text':
            # encoder_layer_sizes (default: [8192, 4096])
            #layer_sizes = opt.encoder_layer_sizes
            print("text layer_sizes:", layer_sizes[0], layer_sizes[-1])
            latent_size_text = opt.latent_size_text
            print("latent_size_text:", latent_size_text)
            #layer_sizes[0] += latent_size_text
            layer_sizes[0] = 8492
            print("text layer_sizes[0]:", layer_sizes[0])
            self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[-1])
            self.fc3 = nn.Linear(layer_sizes[-1], latent_size_text*2)
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.linear_means = nn.Linear(latent_size_text*2, latent_size_text)
            self.linear_log_var = nn.Linear(latent_size_text*2, latent_size_text)
            self.apply(weights_init)

        elif semantics_type == 'image':
            # encoder_layer_sizes (default: [8192, 4096])
            #layer_sizes = opt.encoder_layer_sizes
            print("image layer_sizes:", layer_sizes[0], layer_sizes[-1])
            latent_size_image = opt.latent_size_image
            print("latent_size_image:", latent_size_image)
            layer_sizes[0] += latent_size_image
            print("image layer_sizes[0]:", layer_sizes[0])
            self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[-1])
            self.fc3 = nn.Linear(layer_sizes[-1], latent_size_image*2)
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.linear_means = nn.Linear(latent_size_image*2, latent_size_image)
            self.linear_log_var = nn.Linear(latent_size_image*2, latent_size_image)
            self.apply(weights_init)
        else:
            print("Please indicate which type of semantic embedding is using.")

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x)) #(batch_size, 4096)
        x = self.lrelu(self.fc3(x)) #(batch_size, 600)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


# Decoder/Generator
class Generator(nn.Module):
    def __init__(self, opt, semantics_type=None):
        super(Generator, self).__init__()
        if semantics_type == 'text':
            layer_sizes = opt.decoder_layer_sizes
            latent_size_text = opt.latent_size_text
            input_size = latent_size_text * 2
            self.fc1 = nn.Linear(input_size, layer_sizes[0])
            self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.sigmoid = nn.Sigmoid()
            self.apply(weights_init)

        elif semantics_type == 'image':
            layer_sizes = opt.decoder_layer_sizes
            latent_size_image = opt.latent_size_image
            input_size = latent_size_image * 2
            self.fc1 = nn.Linear(input_size, layer_sizes[0])
            self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.sigmoid = nn.Sigmoid()
            self.apply(weights_init)

        else:
            print("Please indicate which type of semantic embedding is using.")

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z, c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1 * feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x


# conditional discriminator for inductive
class Discriminator_D1(nn.Module):
    def __init__(self, opt, semantics_type=None):
        super(Discriminator_D1, self).__init__()
        if semantics_type == 'text':
            self.fc1 = nn.Linear(opt.resSize + opt.attSize_text, opt.ndh)
            self.fc2 = nn.Linear(opt.ndh, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.apply(weights_init)

        elif semantics_type == 'image':
            self.fc1 = nn.Linear(opt.resSize + opt.attSize_image, opt.ndh)
            self.fc2 = nn.Linear(opt.ndh, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)
            self.apply(weights_init)

        else:
            print("Please indicate which type of semantic embedding is using.")

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h


# Feedback Modules
class Feedback(nn.Module):
    def __init__(self, opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self,x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h


class AttDec(nn.Module):
    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None: 
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        # used at synthesis time and feature transformation
        return self.hidden.detach()

