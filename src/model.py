import torch
from torch import nn

# Define a Multi-Layer Perceptron (MLP) class that inherits from the PyTorch Module class
class MLP(nn.Module):
    # Initialize the MLP with a list of layer widths and an optional dropout probability
    def __init__(self, width_list, p=0):
        # Call the parent class's constructor
        super(MLP, self).__init__()

        # Calculate the depth of the network
        self.depth = len(width_list) - 1

        # Initialize lists for fully connected (fc) layers and batch normalization (bn) layers
        self.fc_list = torch.nn.ModuleList([])
        self.bn_list = torch.nn.ModuleList([])

        # Set the dropout probability
        self.p = p

        # Create the fc and bn layers based on the width list
        for i in range(self.depth):
            self.fc_list.append(nn.Linear(width_list[i], width_list[i + 1]))
            self.bn_list.append(nn.BatchNorm1d(width_list[i]))

        # If dropout is enabled, initialize dropout (do) layers
        if self.p > 0:
            self.do_list = torch.nn.ModuleList([])
            for i in range(self.depth - 1):
                self.do_list.append(nn.Dropout(p=p))

    # Define the forward pass of the MLP
    def forward(self, x):
        # Initialize a list to store the output of each layer
        out_list = []

        # Append the input to the list
        out_list.append(x)

        # Perform the forward pass through each layer except the last
        for i in range(self.depth - 1):
            # Apply fc and bn layers, followed by ReLU activation
            x = self.fc_list[i](self.bn_list[i](x))
            x = x.relu()

            # Append the output to the list
            out_list.append(x)

            # Apply dropout if enabled
            if self.p > 0:
                x = self.do_list[i](x)

        # Perform the forward pass through the last layer without ReLU activation
        x = self.fc_list[-1](self.bn_list[-1](x))

        # Append the output to the list
        out_list.append(x)

        # Return the list of outputs from each layer
        return out_list