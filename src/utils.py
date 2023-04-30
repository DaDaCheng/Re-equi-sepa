import torch

# Function to compute the separation fuzziness 
def compute_logD(X, labels):
    device = X.device
    num_label = int(labels.max() + 1)
    num_feature = X.shape[1]
    X_mean = torch.mean(X, dim=0)
    SSB = torch.zeros((num_feature, num_feature)).to(device) # between-class sum of squares
    SSW = torch.zeros((num_feature, num_feature)).to(device) # within-class sum of squares

    # Calculate SSB and SSW
    for i in range(num_label):
        X_per = X[labels == i, :] # Select the data points corresponding to the current label i
        X_Data_per_label_mean = torch.mean(X_per, dim=0) # Compute the mean feature vector for the data points with label i
        dx = (X_Data_per_label_mean - X_mean).reshape(-1, 1) # Compute the difference between the per-label mean feature vector and the overall mean feature vector
        SSB = SSB + X_per.shape[0] * dx @ dx.T 
        X_Data_per_label_mean_keepdim = torch.mean(X_per, dim=0, keepdim=True)
        ddx = X_per - X_Data_per_label_mean_keepdim
        SSW = SSW + ddx.T @ ddx 
    SSB = SSB / X.shape[0]
    SSW = SSW / X.shape[0]

    # Compute the log determinant ratio
    D = torch.trace(SSW @ torch.linalg.pinv(SSB))
    return torch.log(D)

# Function to compute the logD values for each layer of a model using a dataloader
def compute_logD_list(model, train_gen, size, device):
    model.eval()
    
    depth = model.depth
    # Initialize empty tensors for layer outputs and labels
    Xcat_list = []
    for iii in range(depth + 1):
        Xcat_list.append(torch.tensor([]))
    ycat_list = torch.tensor([])

    # Collect layer outputs and labels using the dataloader, concatenate intermediate output from different batches
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_gen):
            images = images.view(-1, size ** 2).to(device)
            labels = labels.detach().clone().cpu()
            X_ = model(images)
            for iii in range(len(X_)):
                X_[iii] = X_[iii].detach().clone().cpu()
            ycat_list = torch.cat([ycat_list, labels.reshape(-1)], dim=0)
            for k in range(len(X_)):
                Xcat_list[k] = torch.cat([Xcat_list[k], X_[k]], dim=0)
    # Calculate logD values for each layer
    logD_list = []
    for i in range(len(Xcat_list) - 1):
        logD = compute_logD(Xcat_list[i], ycat_list)
        logD_list.append(logD.cpu().numpy())

    return logD_list
