import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import model as module_arch
from utils import load_model
import torch
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class Encoder(nn.Module):

    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super (Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#, return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 1000)
	#self.fc = nn.Linear(num_classes,16) 
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
        x = self.conv1(x)
	
        x = self.bn1(x)
        x = self.relu(x)
	
        x = self.maxpool(x)
	
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
##########################################################################
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dfc3 = nn.Linear(1000, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096,256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256*6*6)
        #self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(64, 1, 12, stride = 4, padding = 4)

    def forward(self,x):#,i1,i2,i3):
        
        x = self.dfc3(x)
        #x = F.relu(x)
        x = F.relu(self.bn3(x))
        
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        #x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        #x = F.relu(x)
        #print(x.size())
        x = x.view(x.shape[0],256,6,6)
        #print (x.size())
        x = F.interpolate(x,scale_factor=2)
        #print x.size()
        x = self.dconv5(x)
        #print x.size()
        x = F.relu(x)
        #print x.size()
        x = F.relu(self.dconv4(x))
        #print x.size()
        x = F.relu(self.dconv3(x))
        #print x.size()        
        x = F.interpolate(x,scale_factor=2)
        #print x.size()        
        x = self.dconv2(x)
        #print x.size()        
        x = F.relu(x)
        x = F.interpolate(x,scale_factor=2)
        #print x.size()
        x = self.dconv1(x)
        #print x.size()
        x = F.interpolate(x,size=[256,1024])
        #x = torch.sigmoid(x)
        #print x
        return x
    
class ResNet_AE(BaseModel):
    def __init__(self):
        super(ResNet_AE,self).__init__()
        self.encoder = Encoder(Bottleneck, [3, 4, 6, 3])
        self.decoder = Decoder()
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#%%
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class autoencoder(BaseModel):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE_MnistModel(BaseModel):
    def __init__(self, latent_dim=20,num_channels=1):
        super(AE_MnistModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 10, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            #nn.AvgPool2d(20),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, kernel_size=5, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(20),
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(10,num_channels,kernel_size=2,stride=2,padding=3,dilation=1)
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE_Apron(BaseModel):
    def __init__(self, latent_dim=20,num_channels=1, filter_list=[(16,5,2),(32,5,2)]):
        super(AE_Apron, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.filter_list = filter_list
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, filter_list[0][0], kernel_size=filter_list[0][1],padding=filter_list[0][2]),
            nn.ReLU(True),
            nn.BatchNorm2d(filter_list[0][0]),
            nn.MaxPool2d(2),
            nn.Conv2d(filter_list[0][0], filter_list[1][0], kernel_size=filter_list[1][1],padding=filter_list[1][2]),
            nn.ReLU(True),
            nn.BatchNorm2d(filter_list[1][0]),
            #nn.AdaptiveAvgPool2d((1, 1))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(filter_list[1][0], filter_list[1][0], kernel_size=5, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(filter_list[1][0]),
            nn.ConvTranspose2d(filter_list[1][0], filter_list[0][0], kernel_size=5, stride=1, dilation=2, bias=False),
            #nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=3,dilation=1)
            # nn.BatchNorm2d(256),
            # nn.ConvTranspose2d(256,num_channels,kernel_size=(7,421),dilation=3,stride=3),
        )
    def forward(self, x):
        x = self.encoder(x)
        #x = self.decoder(x)
        return x


class AE_1layer(BaseModel):
    def __init__(self, num_channels=1,num_filters=16):
        super(AE_1layer, self).__init__()

        self.num_channels = num_channels
        self.num_filters = num_filters
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels,num_filters,kernel_size=(17,5))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters,1,(17,5),1,padding=0,dilation=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

'''
class ApronModel(BaseModel):
    def __init__(self, num_classes=4,num_channels=2):
        super(ApronModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.num_channels, 10, kernel_size=5)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 10, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            #nn.AvgPool2d(20),
            nn.AdaptiveAvgPool2d((1, 1))
            )

    def forward(self, x):
        x = self.encoder(x)

        return x
    

class AE_Apron(nn.Module):
    def __init__(self, latent_dim=20,size=[2,100,1342]):
        super(AE_Apron, self).__init__()
        self.latent_dim = latent_dim
        self.size = size
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(self.size[0]),
            nn.Conv2d(self.size[0], 10, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),
            nn.Conv2d(10, self.latent_dim, kernel_size=5),
            #nn.BatchNorm2d(self.latent_dim),
            #nn.AdaptiveAvgPool2d((1, 1))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 50, kernel_size=5, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(50),
            nn.ConvTranspose2d(50, 100, kernel_size=5, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(100),
            nn.ConvTranspose2d(100,100,kernel_size=3,stride=2,padding=3,dilation=1),
            nn.BatchNorm2d(100),
            #nn.ConvTranspose2d(100,self.size[0],kernel_size=3,dilation=1,stride=2),
            nn.ConvTranspose2d(100,self.size[0],kernel_size=(7,421),dilation=3,stride=3)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x) #,self.size[0],output_size=self.size[1:])
        return x
'''

class FineTuneModel(BaseModel):
    def __init__(self, base_arch='AE_MnistModel',latent_dim=20,num_classes=10,base_ckpt_pth=None):
        super(FineTuneModel, self).__init__()
        self.base_arch = base_arch
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.base_ckpt_pth = base_ckpt_pth
        ae_model = getattr(module_arch.model, base_arch)()
        if base_ckpt_pth:
            ae_model = load_model(ae_model,n_gpu=0,model_ckpt=base_ckpt_pth)
        self.base_model = ae_model.encoder
        self.last_layer = nn.Linear(latent_dim,num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1,self.latent_dim)
        x = self.last_layer(x)
        return F.log_softmax(x, dim=1)
