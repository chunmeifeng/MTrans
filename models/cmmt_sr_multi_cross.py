import torch

from torch import nn
from einops.layers.torch import Rearrange
from models.common import default_conv as conv, Upsampler
from .mca import CrossTransformer


# add by YunluYan


class ReconstructionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReconstructionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        return out


def build_head(args):
    return ReconstructionHead(args.MODEL.INPUT_DIM, args.MODEL.HEAD_HIDDEN_DIM)


class CrossCMMT(nn.Module):

    def __init__(self, args):
        super(CrossCMMT, self).__init__()

        self.head = build_head(args)

        self.head2 = build_head(args)

        x_patch_dim = args.MODEL.HEAD_HIDDEN_DIM * args.MODEL.P1 ** 2
        x_num_patches = (args.INPUT_SIZE // args.MODEL.P1) ** 2

        complement_patch_dim = args.MODEL.HEAD_HIDDEN_DIM * args.MODEL.P2 ** 2
        complement_num_patches = (args.INPUT_SIZE * args.SCALE // args.MODEL.P2) ** 2

        self.x_patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.MODEL.P1,
                      p2=args.MODEL.P1),
        )

        self.complement_patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.MODEL.P2,
                      p2=args.MODEL.P2),
        )

        self.x_pos_embedding = nn.Parameter(torch.randn(1, x_num_patches, x_patch_dim))
        self.complement_pos_embedding = nn.Parameter(torch.randn(1, complement_num_patches, complement_patch_dim))

        self.cross_transformer = CrossTransformer(x_patch_dim, complement_patch_dim, args.MODEL.CTDEPTH,
                                                  args.MODEL.TRANSFORMER_NUM_HEADS,
                                                  args.MODEL.TRANSFORMER_MLP_RATIO)

        self.p1 = args.MODEL.P1
        self.p2 = args.MODEL.P2

        self.up = nn.Sequential(
            Upsampler(conv, args.SCALE, args.MODEL.HEAD_HIDDEN_DIM, act=False),
            conv(args.MODEL.HEAD_HIDDEN_DIM, args.MODEL.OUTPUT_DIM, 3))

        self.tail = nn.Conv2d(args.MODEL.HEAD_HIDDEN_DIM, args.MODEL.OUTPUT_DIM, 1)

    def forward(self, x, complement):
        x = self.head(x)
        complement = self.head2(complement)

        b, _, h, w = x.shape

        _, _, c_h, c_w = complement.shape

        x = self.x_patch_embbeding(x)
        x += self.x_pos_embedding

        complement = self.complement_patch_embbeding(complement)
        complement += self.complement_pos_embedding

        x, complement = self.cross_transformer(x, complement)

        c = int(x.shape[2] / (self.p1 * self.p1))
        H = int(h / self.p1)
        W = int(w / self.p1)

        x = x.reshape(b, H, W, self.p1, self.p1, c)  # b H W p1 p2 c
        x = x.permute(0, 5, 1, 3, 2, 4)  # b c H p1 W p2
        x = x.reshape(b, -1, h, w, )
        x = self.up(x)

        complement = complement.reshape(b, int(c_h/self.p2), int(c_w/self.p2), self.p2, self.p2, int(complement.shape[2]/self.p2/self.p2))
        complement = complement.permute(0, 5, 1, 3, 2, 4)
        complement = complement.reshape(b, -1, c_h, c_w)

        complement = self.tail(complement)

        return x, complement


def build_model(args):
    return CrossCMMT(args)


