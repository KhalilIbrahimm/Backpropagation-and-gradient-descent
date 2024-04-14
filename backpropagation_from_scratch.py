import torch
from torch import nn
from tests_backpropagation import main_test

torch.manual_seed(123)
torch.set_default_dtype(torch.double)

class MyNet(nn.Module):
    def __init__(self, n_l = [2, 3, 2]):
        super().__init__() 
        # number of layers in our network (following Andrew's notations)
        self.L = len(n_l)-1
        self.n_l = n_l
        
        # Where we will store our neuron values
        # - z: before activation function 
        # - a: after activation function (a=f(z))
        self.z = {i : None for i in range(1, self.L+1)}
        self.a = {i : None for i in range(self.L+1)}
        print(self.z)
        # Where we will store the gradients for our custom backpropagation algo
        self.dL_dw = {i : None for i in range(1, self.L+1)}
        self.dL_db = {i : None for i in range(1, self.L+1)}

        # Our activation functions
        self.f = {i : lambda x : torch.tanh(x) for i in range(1, self.L+1)}

        # Derivatives of our activation functions
        self.df = {
            i : lambda x : (1 / (torch.cosh(x)**2)) 
            for i in range(1, self.L+1)
        }
        
        # fully connected layers
        # We have to use nn.ModuleDict and to use strings as keys here to 
        # respect pytorch requirements (otherwise, the model does not learn)
        self.fc = nn.ModuleDict({str(i): None for i in range(1, self.L+1)})
        for i in range(1, self.L+1):
            self.fc[str(i)] = nn.Linear(in_features=n_l[i-1], out_features=n_l[i])
    
    def forward(self, x):
        # Input layer
        self.a[0] = torch.flatten(x, 1)
        
        # Hidden layers until output layer
        for i in range(1, self.L+1):

            # fully connected layer
            self.z[i] = self.fc[str(i)](self.a[i-1])
            # activation
            self.a[i] = self.f[i](self.z[i])

        # return output
        return self.a[self.L]

    def backpropagation(model, y_true, y_pred):
        # Derivative of loss wrt output of the network
        dL_daL = 2 * (y_pred - y_true)
    
        for l in range(model.L, 0, -1):
            #
            dL_dz = dL_daL * model.df[l](model.z[l])
    
            # The @ is a symbol for matrix multiplication in NumPy and PyTorch
            dL_dw = dL_dz.T @ model.a[l-1]
            dL_db = dL_dz.sum(0, keepdim=True)
    
            model.dL_dw[l] = dL_dw
            model.dL_db[l] = dL_db.squeeze(0)
    
            if l > 1:
                dL_daL = dL_dz @ model.fc[str(l)].weight.data
        return None

