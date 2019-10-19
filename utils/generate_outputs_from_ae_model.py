import sys, os
sys.path.append(os.path.abspath("model/"))
sys.path.append(os.path.abspath("data_loader/"))
sys.path.append(os.path.abspath("base/"))
sys.path.append(os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from model import ResNet_AE
from data_loaders import AcousticDataset
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def save_sample(data, outfile):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data)
    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    fig.colorbar(im)
    
    # or we can utilise the get_data_ratio method which is more concise
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    plt.savefig(fname=outfile,dpi=300,format='png')




if __name__ == '__main__':
    home = home = expanduser("~")
    
    save_dir = home + '/acoustic/ae_outputs_batch_16/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = ResNet_AE().to(device)
    
    state = torch.load('/data/acoustic_tmp/ae_acoustic_results_batch_16/models/acoustic_ae/1016_170641/model_best.pth')
    model.load_state_dict(state['state_dict'])
    model.eval()

    #load data
    dataset = AcousticDataset('/gsceph/adapd/acoustic/AA_10/train.pkl')
    
    idxs = np.random.randint(0, len(dataset), 10)
    print(idxs)
    data = []
    for i in idxs:
        img, label = dataset[i]
        name = dataset.data[i][0]
        data.append((name, img, label))
    data = np.array(data)
    print(data[:, 0])

    #generate data
    imgs = np.expand_dims(np.vstack(data[:, 1]), axis=1)
    print(imgs.shape)
    outputs = model(torch.from_numpy(imgs).to(device)).cpu().detach().numpy()

    #plot inputs and outputs
    for index, (full_name, _, label) in enumerate(data):
        name = full_name.split('/')[-1].replace('.pkl', '')
        #print(name)
        save_sample(np.squeeze(imgs[index], axis=0), save_dir + name + '_' + str(label) + '.png')
        save_sample(np.squeeze(outputs[index], axis=0), save_dir + 'output_' + name + '_' + str(label) + '.png')

