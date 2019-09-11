import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def _upsample(x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

class FPA(nn.Module):

    def __init__(self,inplanes,planes=512):
        super(FPA,self).__init__()
        #inplanes = 2048 for RES-5 and 1024 for RES-4
        # Master branch
        self.inplanes = inplanes
        self.conv_master = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(inplanes)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(inplanes)

        # dilated convolution Branchs 
        self.diaconv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,dilation=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.diaconv2 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,dilation=6)
        self.bn2 = nn.BatchNorm2d(planes)
        self.diaconv3 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,dilation=12)
        self.bn3 = nn.BatchNorm2d(planes)

        #self.upsample = nn.ConvTranspose2d(inplanes,inplanes,kernel_size=3,stride=2,padding=1)
        self.upsample1 = nn.ConvTranspose2d(planes,planes,kernel_size=3,stride=1,dilation=3)
        self.upsample2 = nn.ConvTranspose2d(planes,planes,kernel_size=3,stride=1,dilation=6)
        self.upsample3 = nn.ConvTranspose2d(planes,planes,kernel_size=3,stride=1,dilation=12)
        self.conv1x1 = nn.Conv2d(3*planes,inplanes,kernel_size=1,stride=1,bias=False)

        


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)
        x_master = self.relu(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.inplanes, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.relu(x_gpb)

        #branches
        x_1 = self.diaconv1(x)
        x_1 = self.bn1(x_1)
        x_1 = self.relu(x_1)

        x_2 = self.diaconv2(x)
        x_2 = self.bn2(x_2)
        x_2 = self.relu(x_2)

        x_3 = self.diaconv3(x)
        x_3 = self.bn3(x_3)
        x_3 = self.relu(x_3)

        x_1 = self.upsample1(x_1)
        x_2 = self.upsample2(x_2)
        x_3 = self.upsample3(x_3)

        x_cat = torch.cat((x_3,x_2,x_1),1)
        x_cat = self.conv1x1(x_cat)
        #out = self.upsample(x_cat,output_size=x_master.size())

        out = x_cat * x_master
        out = out + x_gpb
        out = self.relu(out)

        return out #feature size is 1024

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=0, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.convreduce = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        #self.upsample = nn.ConvTranspose2d(channels_high,channels_low,kernel_size=3,stride=2,padding=1) 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        #fms_low_mask = self.relu(fms_low_mask)
        
        fms_att = fms_low_mask * fms_high_gp
        fms_high_upsample = _upsample(fms_high,fms_low_mask)
        fms_high_upsample = self.convreduce(fms_high_upsample)
        #fms_high_upsample =  self.upsample(fms_high,output_size=fms_low_mask)

        out = fms_high_upsample + fms_att
        out = self.relu(out)
        return out #same feature size as passed Res stage 





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, scale=1):
        self.inplanes = 64
        self.scale = scale
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_layer(block, 64, layers[0])
        self.stage3 = self._make_layer(block, 128, layers[1], stride=2)
        self.stage4 = self._make_layer(block, 256, layers[2], stride=2)
        #top down
        self.fpa = FPA(inplanes=1024)
        self.gau_block1 = GAU(1024,512)
        self.gau_block2 = GAU(512,256)
        self.conv2 = nn.Conv2d(1792,1024,kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        #self.stage5 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        h = self.stage2(h)
        res2 = h
        h = self.stage3(h)
        res3 = h
        h = self.stage4(h)
        res4 = h

        p4 = self.fpa(res4)#1024
        p3 = self.gau_block1(p4,res3)#512
        p2 = self.gau_block2(p3,res2)#256

        p3 = _upsample(p3, p2)
        p4 = _upsample(p4, p2)

        out = torch.cat((p2,p3,p4),1)

        out = self.conv2(out)
        out = self.relu2(self.bn2(out))

        #out = self.stage5(out)
        out = self.conv3(out)
        out = self.relu3(self.bn3(out))
        out = self.conv4(out)

        out = _upsample(out,x,scale=self.scale)
        return out    

def resPAnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resPAnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resPAnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


def resPAnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet101'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model

def resPAnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet152'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model