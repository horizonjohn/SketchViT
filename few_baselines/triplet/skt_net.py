import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchANetSBIR(nn.Module):
    def __init__(self):
        super(SketchANetSBIR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=15, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc6 = nn.Linear(2304, 512)
        self.fc7 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # [10, 2304]
        x = F.relu(self.fc6(x))
        x = F.normalize(self.fc7(x), p=2, dim=1)
        return x


def spatial_softmax(fm):
    fm_shape = fm.shape
    n_grids = fm_shape[2] ** 2
    # transpose feature map
    fm = fm.permute(0, 2, 3, 1)
    t_fm_shape = fm.shape
    fm = fm.reshape((-1, n_grids))
    # apply softmax
    prob = F.softmax(fm, dim=1)
    # reshape back
    prob = prob.reshape(t_fm_shape)
    prob = prob.permute(0, 3, 1, 2)
    return prob


class AttentionNet(nn.Module):
    def __init__(self, pool_method='sigmoid'):
        super(AttentionNet, self).__init__()
        assert(pool_method in ['sigmoid', 'softmax'])
        self.pool_method = pool_method
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.pool_method == 'sigmoid':
            x = F.sigmoid(self.conv2(x))
        else:
            x = self.conv2(x)
            x = spatial_softmax(x)
        return x


class SketchANetDSSA(nn.Module):
    def __init__(self, trainable=False):
        super(SketchANetDSSA, self).__init__()
        self.conv1_s1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=15, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2_s1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3_s1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4_s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv5_s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3)
        self.attentionNet = AttentionNet(pool_method='softmax')
        self.fc6_s1 = nn.Linear(in_features=2304, out_features=512)
        self.fc7_sketch = nn.Linear(in_features=512, out_features=256)

    def forward(self, x):
        x = F.relu(self.conv1_s1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_s1(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_s1(x))
        x = F.relu(self.conv4_s1(x))
        x = self.conv5_s1(x)
        x = self.pool3(x)  # [10, 256, 3, 3]
        att_mask = self.attentionNet(x)
        att_map = x * att_mask
        att_f = x + att_map
        attended_map = torch.sum(att_f, dim=[2, 3])
        attended_map = F.normalize(attended_map, dim=1)
        att_f = att_f.view(att_f.size(0), -1)  # [10, 2304]
        fc6 = F.relu(self.fc6_s1(att_f))
        fc7 = self.fc7_sketch(fc6)
        fc7 = F.normalize(fc7, dim=1)
        final_feature_map = torch.cat((fc7, attended_map), dim=1)
        return final_feature_map


if __name__ == "__main__":
    from torchinfo import summary
    model = SketchANetDSSA()
    print('loaded')
    print(model)

    print(summary(model, (1, 3, 299, 299)))

    embedding = model(torch.randn(10, 3, 299, 299).cuda())
    print(embedding.shape)
    # for p in model.parameters():
    #    print(p.requires_grad, p.shape)
