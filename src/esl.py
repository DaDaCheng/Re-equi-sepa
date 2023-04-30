import numpy as np
import torch
from torch import nn
import sys
from utils import compute_logD_list
from model import MLP
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import trange




class ESL(object):
    """
    class provides an implementation of ESL(equi-separation law) in deep learning
    for various datasets and supports customization of the underlying model.

    Args:
        dataset (str, optional): The name of the dataset to be used. Supported datasets are 
            'MNIST', 'FashionMNIST', 'FakeData', and 'CIFAR10'. If a custom dataset is used, 
            provide the dataset object instead of a string. Defaults to 'MNIST'.
        data_length (int, optional): The number of data points to be used from the dataset. 
            Defaults to 1000.
        batch_size (int, optional): The batch size for training the model. Defaults to 100.
        data_size (int, optional): The size to which the input images should be resized. 
            Defaults to 10.

    Attributes:
        train_loader (DataLoader): DataLoader object for loading and batching training data.
        model (MLP): The underlying neural network model.

    Example:
        # Instantiate the ESL class with the MNIST dataset
        esl = ESL(dataset='MNIST', data_length=1000, batch_size=100, data_size=10)

        # Set the model architecture
        esl.set_model(hidden_dim=100, depth=6, width_list=None, p=0.05, device='cpu')

        # Train the model
        esl.train(lr=1e-3, num_epochs=100, stop_loss=1e-3, opt='Adam')

        # Compute the separation values
        logD_list = esl.compute_separation()

        # Plot the separation values
        esl.plot_separation(logD_list)
    """
    
    
    
    def __init__(self, dataset='MNIST', data_length=1000, batch_size = 100, data_size=10):
        self.dataset=dataset
        self.dataset_length=data_length
        self.data_size=data_size
        # Load and preprocess the dataset
        if dataset=='MNIST':
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=data_size)
            ])
            train_data = datasets.MNIST(root = './data', train = True,
                                    transform = transform, download = True)
            self.num_classes= len(train_data.classes)
            # Split the dataset into a smaller subset
            train_data, _ = torch.utils.data.random_split(train_data, [data_length, 60000-data_length])
        if dataset=='FashionMNIST':
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=data_size)
            ])
            train_data = datasets.FashionMNIST(root = './data', train = True,
                                    transform = transform, download = True)
            self.num_classes= len(train_data.classes)
            train_data, _ = torch.utils.data.random_split(train_data, [data_length, 60000-data_length])

        if dataset=='FakeData':
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0,1),
            ])
            train_data = datasets.FakeData(size=data_length,image_size=[data_size,data_size],num_classes=10,transform = transform)
            self.num_classes= len(train_data.classes) 
        if dataset=='CIFAR10':
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Normalize(0,1),
                transforms.Resize(size=data_size),
            ])
            train_data = datasets.CIFAR10(root = './data', train = True,
                                    transform = transform, download = True)
            self.num_classes= len(train_data.classes)
            train_data, _ = torch.utils.data.random_split(train_data, [data_length, 50000-data_length])
        # Custom dataset
        if dataset not in ['MNIST','FashionMNIST','FakeData','CIFAR10']:
            train_data=dataset
            self.num_classes= len(train_data.classes)
        self.train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                                        batch_size = batch_size,
                                                        shuffle = True,
                                                        pin_memory=True)
    def set_model(self,hidden_dim=100,depth=6,width_list=None,p=0,device='cpu'):
        if width_list==None:
            width_list=[hidden_dim for i in range(depth)]
            width_list.append(self.num_classes)
        else:
            width_list.append(self.num_classes)
        self.model=MLP(width_list,p=p).to(device)
    def train(self,lr=1e-3,num_epochs=100,stop_loss=1e-3,opt='Adam'):
        if opt=='Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if opt=='SGD':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if opt not in ['Adam','SGD']:
            optimizer=opt
        loss_func=nn.CrossEntropyLoss()
        tbar = trange(num_epochs, desc='Batch loss', leave=True)
        self.model.train()
        for epoch in tbar:
            batch_loss=0
            num_data=0
            for i ,(images,labels) in enumerate(self.train_loader):
                images= images.view(-1,self.data_size**2).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out= self.model(images)
                loss = loss_func(out[-1], labels)
                loss.backward()
                optimizer.step()
                batch_loss+=loss.item()*labels.numel()
                num_data+=labels.numel()
            batch_loss=batch_loss/num_data
            tbar.set_description("Train loss={:.2e}".format(batch_loss), refresh=True)
            if batch_loss<stop_loss:
                break
    def compute_separation(self):
        device=self.model.fc_list[0].weight.device
        logD_list=compute_logD_list(self.model,self.train_loader,self.data_size,device)
        return logD_list
    def plot_separation(self,logD_list):
        logD_list=np.array(logD_list)
        import matplotlib.pyplot as plt
        from scipy import stats
        color_list_two=[(7/255,7/255,7/255),(255/255,59/255,59/255)]
        index_list=np.arange(len(logD_list))
        slope, intercept, r_value, p_value, std_err = stats.linregress(index_list,logD_list)
        r=stats.pearsonr(index_list,logD_list)[0]
        print("Pearson correlation coefficient:{:.3}".format(r))
        x_0=[0,len(index_list)-1]
        y_0=[intercept,intercept+(len(index_list)-1)*slope]
        plt.rcParams["figure.figsize"] = (7,5)
        plt.rcParams.update({'font.size': 18})
        plt.plot(index_list,logD_list,'.',markersize=20,color=color_list_two[1])
        plt.plot(x_0,y_0,'-',color=color_list_two[0],linewidth=3,label="r={:.3}".format(r))
        plt.ylabel(r'$\log(D_i)$')
        plt.legend()
        plt.xlabel('Layer index: i')
        plt.show()
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    esl=ESL(dataset='MNIST',data_size=10)
    esl.set_model(hidden_dim=100,depth=6,p=0.05,device=device)
    esl.train()
    logD_list=esl.compute_separation()
    esl.plot_separation(logD_list)
