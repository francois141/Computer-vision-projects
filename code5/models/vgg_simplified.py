import torch
import torch.nn as nn
import math

class ConvReLUPool(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvReLUPool,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)

        self.maxPool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

    def forward(self,x):
        return self.maxPool(self.relu(self.conv(x)))

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()

        self.linear1 = nn.Linear(512,128)
        self.linear2 = nn.Linear(128,10)

        self.drop = nn.Dropout(0.2)

        self.relu = nn.ReLU()

    def forward(self,x):
        return self.linear2(self.drop(self.relu(self.linear1(x))))

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        self.conv_block1 = ConvReLUPool(3,64)
        self.conv_block2 = ConvReLUPool(64,128)
        self.conv_block3 = ConvReLUPool(128,256)
        self.conv_block4 = ConvReLUPool(256,512)
        self.conv_block5 = ConvReLUPool(512,512)

        self.classifier = Classifier()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        score = self.classifier(x.squeeze())
        return score

