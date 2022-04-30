import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=0, bias=False)
        self.bn1 = nn.LayerNorm([32, 49, 49])

        self.actvn = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actvn(x)

        return self.flatten(x)


if __name__ == '__main__':
    model = SampleModel()
    summary(model.cuda(), [[1, 51, 51]])
