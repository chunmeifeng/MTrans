import torch

from torch import nn
from models.transformer import build_transformer
from einops.layers.torch import Rearrange
from models.common import default_conv as conv, Upsampler

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


class CMMT(nn.Module):

    def __init__(self, args):
        super(CMMT, self).__init__()

        self.head = build_head(args)

        patch_dim = args.MODEL.HEAD_HIDDEN_DIM* args.MODEL.PATCH_SIZE ** 2
        num_patches = (args.INPUT_SIZE * args.SCALE // args.MODEL.PATCH_SIZE) ** 2

        self.transformer = build_transformer(args, patch_dim)

        self.patch_embbeding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=args.MODEL.PATCH_SIZE,
                         p2=args.MODEL.PATCH_SIZE),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, patch_dim))


        self.p1 = args.MODEL.PATCH_SIZE
        self.p2 = args.MODEL.PATCH_SIZE

        self.up = nn.Sequential(
            Upsampler(conv, args.SCALE, args.MODEL.HEAD_HIDDEN_DIM, act=False),
            conv(args.MODEL.HEAD_HIDDEN_DIM, args.MODEL.HEAD_HIDDEN_DIM, 3))

        self.tail = nn.Conv2d(args.MODEL.HEAD_HIDDEN_DIM, args.MODEL.OUTPUT_DIM, 1)

    def forward(self, x):

        x = self.head(x)

        x = self.up(x)

        b, _, h, w = x.shape

        x= self.patch_embbeding(x)

        x += self.pos_embedding

        x = self.transformer(x)  # b HW p1p2c

        c = int(x.shape[2]/(self.p1*self.p2))
        H = int(h/self.p1)
        W = int(w/self.p2)

        x = x.reshape(b, H, W, self.p1, self.p2, c)  # b H W p1 p2 c
        x = x.permute(0, 5, 1, 3, 2, 4)  # b c H p1 W p2
        x = x.reshape(b, -1, h, w,)
        x= self.tail(x)
        return x

def build_model(args):
    return CMMT(args)








