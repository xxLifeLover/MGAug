import torch.nn as nn
from net.net_utils import RWLinear

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.use_maxpool = use_maxpool

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if use_maxpool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, args, depth):
        super().__init__()
        self.args = args
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.out_features = args.num_way

        self.encoder = []
        for i in range(depth):
            in_channels = 3 if i == 0 else 64
            self.encoder.append(ConvBlock(in_channels, self.hidden_channels, use_maxpool=(i < 4)))
        self.encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*self.encoder)

        self.classifier = RWLinear(1., self.hidden_channels * 5 * 5, self.out_features, us=[True, False])
        self.init_params()

    def init_params(self):
        for k, v in self.named_parameters():
            if ('Conv' in k) or ('Linear' in k):
                if ('weight' in k):
                    nn.init.kaiming_uniform_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('Batch' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
        return None

    def forward(self, x):
        x = self.encoder(x)  # (N, 3, 80, 80) -> (N, 64, 5, 5)
        x = self.classifier(x) # (N, 5)
        return x