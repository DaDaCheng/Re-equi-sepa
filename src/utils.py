import torch

def compute_logD(X,labels):
    device=X.device
    number_label=int(labels.max()+1)
    numb_feature=X.shape[1]
    X_mean=torch.mean(X,dim=0)
    SSB=torch.zeros((numb_feature,numb_feature)).to(device)
    SSW=torch.zeros((numb_feature,numb_feature)).to(device)
    for i in range(number_label):        
        X_per=X[labels==i,:]
        X_Data_per_label_mean=torch.mean(X_per,dim=0)
        dx=(X_Data_per_label_mean-X_mean).reshape(-1,1)
        SSB=SSB+X_per.shape[0]*dx@dx.T
        X_Data_per_label_mean_keepdim=torch.mean(X_per,dim=0,keepdim=True)
        ddx=X_per-X_Data_per_label_mean_keepdim
        SSW=SSW+ddx.T@ddx
    SSB=SSB/X.shape[0]
    SSW=SSW/X.shape[0]
    D=torch.trace(SSW@torch.linalg.pinv(SSB))
    return torch.log(D)

def compute_logD_list(model,train_gen,size,device):
    Xcat_list=[]
    model.eval()
    depth=model.depth
    for iii in range(depth+1):
        Xcat_list.append(torch.tensor([]))
    ycat_list=torch.tensor([])
    with torch.no_grad():
        for i ,(images,labels) in enumerate(train_gen):
            model.eval()
            images = images.view(-1,size**2).to(device)
            labels = labels.detach().clone().cpu()
            X_= model(images)
            for iii in range(len(X_)):
                X_[iii]=X_[iii].detach().clone().cpu()
            ycat_list=torch.cat([ycat_list,labels.reshape(-1)],dim=0)
            for k in range(len(X_)):
                Xcat_list[k]=torch.cat([Xcat_list[k],X_[k]],dim=0)
    
    logD_list=[]
    for i in range(len(Xcat_list)-1):
        logD=compute_logD(Xcat_list[i],ycat_list)
        logD_list.append(logD.cpu().numpy())
    return logD_list