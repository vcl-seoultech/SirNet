import torch
import models.DenseNet
import models.AdaINGenerator

"""
This code converted from 
https://github.com/EscVM/Efficient-CapsNet/blob/3d8d51e7a7c777c59eb8db9dae8bb14d72d5de03/utils/layers.py#L90
"""

gen = {
  'activ': 'lrelu',                   # activation function style [relu/lrelu/prelu/selu/tanh]
  'dec': 'basic',                     # [basic/parallel/series]
  'dim': 16,                        # number of filters in the bottommost layer
  'dropout': 0,                     # use dropout in the generator
  'id_dim': 2048,                   # length of appearance code
  'mlp_dim': 512,                   # number of filters in MLP
  'mlp_norm': 'none',                 # norm in mlp [none/bn/in/ln]
  'n_downsample': 5,                # number of downsampling layers in content encoder
  'n_res': 4,                       # number of residual blocks in content encoder/decoder
  'non_local': 0,                   # number of non_local layer
  'pad_type': 'zero',              # padding type [zero/reflect]
  'tanh': 'false',                    # use tanh or not at the last layer
  'init': 'kaiming'                  # initialization [gaussian/kaiming/xavier/orthogonal]
}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class PrimaryCaps(torch.nn.Module):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    Properties of each capsule s_n are exatracted using a 2D depthwise convolution.

    ...

    Attributes
    ----------
    F: int
        depthwise conv number of features
    K: int
        depthwise conv kernel dimension
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    s: int
        depthwise conv strides
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """

    def __init__(self, F, K, N, D, s=1):
        super().__init__()
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s

        self.Conv2D = torch.nn.Conv2d(self.F, self.F, self.K, self.s, groups=self.F)
        self.built = True

    def forward(self, inputs):
        x = self.Conv2D(inputs)
        x = torch.reshape(x, (inputs.shape[0], self.N, self.D))
        x = Squash()(x)

        return x


class Squash(torch.nn.Module):
    """
    Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.

    ...

    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression

    Methods
    -------
    call(s)
        compute the activation from input capsules
    """

    def __init__(self, eps=1e-20):
        super().__init__()
        self.eps = eps

    def forward(self, s):
        # n = torch.norm(s, dim=-1, keepdim=True)
        # return (1 - 1 / (torch.exp(n) + self.eps)) * (s / (n + self.eps))
        squared_norm = (s ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * s / torch.sqrt(squared_norm)


class FCCaps(torch.nn.Module):
    """
    Fully-connected caps layer. It exploites the routing mechanism, explained in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing',
    to create a parent layer of capsules.

    ...

    Attributes
    ----------
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    kernel_initilizer: str
        matrix W initialization strategy

    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """

    def __init__(self, input_N, input_D, N, D, kernel_initializer=torch.nn.init.kaiming_normal_):
        super().__init__()
        self.N = N
        self.D = D
        self.kernel_initializer = kernel_initializer

        input_D = torch.tensor(input_D)

        self.W = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty([self.N, input_N, input_D, self.D])))
        self.b = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty([self.N, input_N, 1])))
        self.built = True

    def forward(self, inputs):
        u = torch.einsum('...ji,kjiz->...kjz', inputs, self.W)  # u shape=(None,N,H*W*input_N,D)

        c = torch.einsum('...ij,...kj->...i', u, u)[..., None]  # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c / (self.D ** 0.5)
        c = torch.softmax(c, dim=1)  # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.b
        s = torch.sum(torch.multiply(u, c), dim=-2)  # s shape=(None,N,D)
        v = Squash()(s)  # v shape=(None,N,D)

        return v


class EmbeddingCapsNet(torch.nn.Module):
    def __init__(self, num_classes=490, primary_dim=(2560, 8), digit_dim=(2048, 4), ratio: int = 16, cuda=False):
        super().__init__()

        self.ratio = ratio

        self.dense_feature = models.DenseNet.densenet121(pretrained=True)

        self.classifier_activation = torch.nn.Linear(1024, num_classes)
        torch.nn.init.kaiming_normal_(self.classifier_activation.weight)

        # self.PrimaryCaps = PrimaryCaps(128, 9, 16, 8)
        self.PrimaryCaps = PrimaryCaps(1024, 7, *primary_dim)
        self.dropout = DropCapsule(0.5, cuda=cuda)

        self.digit_caps_fore = FCCaps(primary_dim[0] // ratio * (ratio - 1), primary_dim[1], digit_dim[0] // ratio * (ratio - 1), digit_dim[1])
        self.digit_caps_back = FCCaps(primary_dim[0] // ratio, primary_dim[1], digit_dim[0] // ratio, digit_dim[1])

        self.classifier_fore = torch.nn.Linear(digit_dim[0] // ratio * (ratio - 1), num_classes)
        torch.nn.init.kaiming_normal_(self.classifier_fore.weight)
        self.classifier_back = torch.nn.Linear(digit_dim[0] // ratio, num_classes)
        torch.nn.init.kaiming_normal_(self.classifier_back.weight)

        self.reconstructor = Generator(digit_dim[0] // ratio * (ratio - 1) * digit_dim[1], digit_dim[0] // ratio * digit_dim[1], digit_dim[0] // ratio * (ratio - 1))

    def forward(self, x):
        x = self.dense_feature.features(x)

        mean_pool = torch.reshape(torch.nn.AvgPool2d(x.shape[2:4])(x), x.shape[0:2])

        primary_caps = self.PrimaryCaps(x)
        drop_caps = self.dropout(primary_caps)

        digit_caps_fore = self.digit_caps_fore(drop_caps[:, :drop_caps.shape[1] // self.ratio * (self.ratio - 1)])
        digit_caps_back = self.digit_caps_back(drop_caps[:, drop_caps.shape[1] // self.ratio * (self.ratio - 1):])

        features_fore = l2(digit_caps_fore)
        features_back = l2(digit_caps_back)

        return [self.classifier_fore(features_fore), self.classifier_back(features_back), self.classifier_activation(mean_pool)], features_fore - 0.5, [torch.cat((digit_caps_fore, digit_caps_back), dim=1), x.clone().detach()]


class DropCapsule(torch.nn.Module):
    def __init__(self, ratio=0.5, cuda=False):
        """
        Dropout for capsules.
        :param ratio: dropout ratio
        :param cuda: True if use Gpus else False
        """

        super().__init__()

        self.gpu = cuda
        self.dropout = torch.nn.Dropout(ratio)

    def forward(self, capsules):
        """
        When input capsules, it returns dropouted capsule
        :param capsules: Capsule layer. Its shape must be (bach size, The number of capsules, capsule dimension)
        :return: Dropouted capsules
        """

        b, n, d = capsules.shape

        mask = torch.ones((b, n)).cuda() if self.gpu else torch.ones((b, n))
        mask = self.dropout(mask)
        mask = torch.autograd.Variable(mask)

        return capsules * torch.stack((mask, ) * d, dim=2)


def l2(capsules):
    """
    Calculate capsules norms
    (batch size, the number of capsules, capsule's dimension) -> (batch size, the number of capsules)
    :param capsules: Input capsules
    :return: l2 norm of capsules
    """

    return torch.sqrt(torch.sum(capsules ** 2, dim=2))


class Generator(torch.nn.Module):
    def __init__(self, in_channel: int = 7680, back_channel: int = 1024, feature_n: int = 1920):
        """
        Initialize
        :param in_channel: length of id feature
        :param feature_n: the number of id capsules
        """

        super().__init__()

        self.main = models.AdaINGenerator.AdaINGen(back_channel, gen, fp16=False)
        self.feature_n = feature_n

        fcl = list()
        linear1 = torch.nn.Linear(in_channel, 8192)
        torch.nn.init.kaiming_normal_(linear1.weight)
        fcl.append(linear1)
        fcl.append(torch.nn.ReLU(inplace=True))
        self.fcl = torch.nn.Sequential(*fcl)

    def forward(self, inputs):
        fore = torch.reshape(inputs[:, :self.feature_n], (inputs.shape[0], -1))
        back = torch.reshape(inputs[:, self.feature_n:], (inputs.shape[0], -1, 1, 1))

        back = torch.cat((back, back), dim=2)
        back = torch.cat((back, back), dim=3)

        feature = self.fcl(fore)
        recon = self.main.decode(back, feature)
        return torch.reshape(recon[0], (inputs.shape[0], -1)), torch.reshape(recon[1], (inputs.shape[0], -1))
