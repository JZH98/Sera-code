# This is the networks script
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device('cuda:1')

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(LinearBlock, self).__init__()
        if activation is True:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class Patcher(nn.Module):
    def __init__(self, patch_size, stride, in_chan, out_dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        if out_dim == int(in_chan*patch_size[0]*patch_size[1]):
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(int(in_chan * patch_size[0] * patch_size[1]), out_dim, bias=False),
            )
        self.to_patch = Rearrange("b l n -> b n l")

    def forward(self, x):
        # x: b, k, c, l
        x = self.unfold(x)
        x = self.to_patch(x)
        x = self.to_out(x)
        return x


# Multiple Multi-stage Decoder VAE (MMDVAE)
class MMDVAE(nn.Module):

    def __init__(self, dimx, dimz, n_sources=3, device='gpu', variational=True):
        super(MMDVAE, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.device = device
        self.variational = variational

        chans = (128, 64, self.dimz)

        self.out_z = nn.Linear(chans[-1], 2 * self.n_sources * self.dimz)

        self.Encoder = nn.Sequential(
            LinearBlock(self.dimx, chans[0]),
            LinearBlock(chans[0], chans[1]),
            LinearBlock(chans[1], chans[2])
        )

        self.DecoderLv1 = nn.Sequential(
            LinearBlock(self.dimz, chans[-1]),
            LinearBlock(chans[-1], self.dimz),
        )

        self.DecoderLv2 = nn.Sequential(
            LinearBlock(chans[2], chans[1]),
            LinearBlock(chans[1], chans[0]),
            LinearBlock(chans[0], self.dimx, activation=False),
        )

    def encode(self, x):
        d = self.Encoder(x)
        dz = self.out_z(d)
        mu = dz[:, ::2]
        logvar = dz[:, 1::2]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #d = self.DecoderLv1(z.view(-1, self.dimz))
        recon_separate = torch.sigmoid(z).view(-1, self.n_sources, self.dimz)
        b, _, _ = recon_separate.size()
        recon_separate = rearrange(recon_separate, 'b s z -> (b s) z')
        recon_separate = self.DecoderLv1(recon_separate)
        recon_separate = rearrange(recon_separate, '(b s) z -> b s z', b=b)
        recon_x = recon_separate.sum(1)
        recon_x = self.DecoderLv2(recon_x)
        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.variational is True:
            z = self.reparameterize(mu, logvar)
            recon_x, recons = self.decode(z)
        else:
            recon_x, recons = self.decode(mu)
        return recon_x, mu, logvar, recons


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class encoderTSc(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=padding),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(encoderTSc, self).__init__()
        self.inception_window = [0.5, 0.375, 0.25, 0.125]
        self.pool = 2
        if 0.5*sampling_rate % 2 == 0:
            kernel_size = (1, int(0.5 * sampling_rate + 1))
        else:
            kernel_size = (1, int(0.5 * sampling_rate))
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, int(num_T * 1), (1, int(self.inception_window[0] * sampling_rate + 1)),
                                         1, self.pool, self.get_padding(int(self.inception_window[0] * sampling_rate + 1)))
        self.Tception2 = self.conv_block(1, int(num_T * 1), (1, int(self.inception_window[1] * sampling_rate + 1)),
                                         1, self.pool, self.get_padding(int(self.inception_window[1] * sampling_rate + 1)))
        self.Tception3 = self.conv_block(1, int(num_T * 1), (1, int(self.inception_window[2] * sampling_rate + 1)),
                                         1, self.pool, self.get_padding(int(self.inception_window[2] * sampling_rate + 1)))
        self.Tception4 = self.conv_block(1, int(num_T * 1), (1, int(self.inception_window[3] * sampling_rate + 1)),
                                         1, self.pool, self.get_padding(int(self.inception_window[3] * sampling_rate + 1)))
        self.Tfusion = self.conv_block(num_T * 4, num_T, (1, 1), 1, int(self.pool * 0.5))

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.5))
        # self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
        #                                  int(self.pool))
        # self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        # self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            # nn.Linear(22, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # nn.Linear(hidden, hidden * 4)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=1)
        out = self.Tfusion(out)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        out = self.BN_s(out_)
        out = self.fc(out)
        return out
    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

class SERA(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=padding),
            nn.LeakyReLU(),
            )

    def __init__(self, num_classes, input_size, sampling_rate, num_T, patch_size, patch_stride,
                dropout_rate, pool=2, dimz=32, m=3, transformer_depth=2, num_head=16):
        # input_size: EEG frequency x channel x datapoint
        super(SERA, self).__init__()
        self.pool = pool
        self.channel = input_size[1]
        self.num_S = 32
        if 0.5*sampling_rate % 2 == 0:
            kernel_size = (1, int(0.5 * sampling_rate + 1))
        else:
            kernel_size = (1, int(0.5 * sampling_rate))
        self.inception_window = [0.5, 0.25, 0.125]
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool, self.get_padding(kernel_size[-1]))
        self.Sception1 = self.conv_block(num_T, num_T, (int(input_size[1]), 1), 1, int(self.pool))
        self.BN_t = nn.BatchNorm2d(num_T)
        # self.BN_s = nn.BatchNorm2d(num_T)
        self.encodertsc = encoderTSc(num_classes, input_size, sampling_rate, num_T, num_S=num_T, hidden=dimz * 2, dropout_rate=dropout_rate)

        self.encoders = nn.Sequential(
            Conv2dWithConstraint(input_size[0], num_T, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(num_T, num_T, (input_size[-2], 1), padding=0, max_norm=2),
            nn.BatchNorm2d(num_T),
            nn.ELU(),
            nn.MaxPool2d((1, self.pool), stride=(1, self.pool))
        )
        self.patch_size = [1, patch_size]
        self.patch_stride = [1, patch_stride]
        assert (input_size[-1]/self.pool) % patch_size == 0, "The time dimension cannot be divided by patch size evenly! "
        self.in_chan = num_T
        self.out_dim = int(num_T * self.patch_size[0] * self.patch_size[1])
        self.patcher = Patcher(self.patch_size, self.patch_stride, self.in_chan, self.out_dim)

        self.seq = int((input_size[-1]/self.pool - patch_size) // patch_stride + 1)
        self.dimx = int(num_T * self.patch_size[0] * self.patch_size[1])
        self.dimz = dimz
        self.m = m
        self.vae = MMDVAE(self.dimx, self.dimz, self.m)

        self.domain_classifier = nn.Sequential(
            nn.Linear(int(self.seq * self.m * self.dimz), int(self.seq * self.m)),
            nn.ReLU(),
            nn.Linear(int(self.seq * self.m), 2)
        )# 2 classes: source or target= DomainClassifier()
        self.encodert = nn.Sequential(
            nn.Linear(int(self.m * self.dimz), int(self.dimz)),
            nn.ReLU(),
            nn.Linear(int(self.dimz), int(self.dimz))
        )
        self.transformer_depth = transformer_depth
        self.transformer = Transformer(self.dimz, self.transformer_depth, num_head, self.dimz, self.dimz, dropout_rate)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.seq * self.dimz), num_classes))
        self.decoder1 = nn.Sequential(
            nn.Linear(int(input_size[-1]), int(self.m * self.dimz)),
            nn.ReLU(),
        )


    def forward(self, x, alpha=1.0):  ###added alpha
        y = self.encodertsc(x)  # (b, 1, c, t) -> (b, k, 1, t1)
        y = self.patcher(y)  # (b, k, 1, t1) -> (b, seq, h)
        res = self.decoder1(y)
        y = rearrange(y, 'b seq h -> (b seq) h')
        y_rec, mu, logvar, z_source = self.vae(y)
        z_s_d = rearrange(z_source, 'b s d -> b (s d)')
        z_t = rearrange(z_s_d, '(b s) h -> b s h', s=self.seq)

        z_domain = rearrange(z_t, 'b s h -> b (s h)', s=self.seq)
        rf = ReverseLayerF.apply(z_domain, alpha)  ###reverse_features
        dss = self.domain_classifier(rf)  # (b, h1) -> (b, 2)    loss_d

        dta = self.encodert(z_t) # + res1  # (b, seq, h2) -> (b, seq, h3) loss_ta
        out = self.transformer(dta) + dta   #
        out_tsne = out  # for tsne figure
        out = out.view(x.size(0), -1)
        out = self.fc(out)  # label  (B,T,H5)->(B,2)  loss_ce

        return y, y_rec, z_source, out, dss, dta, out_tsne

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def covariance_matrix(X):
    # X is of shape (batch_size, n, m)
    # Center the data
    X_centered = X - X.mean(dim=1, keepdim=True)
    # Compute covariance matrix for each sample in the batch
    cov_matrix = torch.matmul(X_centered.transpose(1, 2), X_centered) / (X_centered.size(1) - 1)
    return cov_matrix


def temporal_alignment_loss(P, Q):
    # P and Q: b, seq, h
    # Compute batch covariance matrices
    cov_P = covariance_matrix(P)
    cov_Q = covariance_matrix(Q)

    diff = cov_P - cov_Q
    # Compute the Frobenius norm of the differences for each pair
    loss = torch.norm(diff, p='fro', dim=(1, 2))
    # Average loss over the batch
    loss = torch.mean(loss)
    return loss


if __name__ == "__main__":
    data = torch.ones((16, 1, 32, 512))
    net = SERA(num_classes=2, input_size=(1, 32, 512), sampling_rate=128, num_T=32, patch_size=16, patch_stride=8,
               dropout_rate=0.25, pool=2, dimz=32)
    print(net)
    print(count_parameters(net))

    y, y_rec, z_source, out, dss, sdta, out_tsne = net(data)
    tdta = sdta
    if tdta.size()[0] == sdta.size()[0]:
        loss_ta = temporal_alignment_loss(tdta, sdta)
    else:
        loss_ta = 0

    print('Done')
