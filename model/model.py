import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import model as module_arch
from utils import load_model

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
