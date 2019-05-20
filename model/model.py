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
    def __init__(self, latent_dim=10):
        super(AE_MnistModel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
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
            nn.ConvTranspose2d(10,1,kernel_size=2,stride=2,padding=3,dilation=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
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
