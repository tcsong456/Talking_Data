from torch import nn

class NN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_size_1=512,
                 hidden_size_2=128,
                 output_size=12):
        super().__init__()
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax()
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.2)
        self.ff1 = nn.Linear(input_dim,hidden_size_1)
        self.ff2 = nn.Linear(hidden_size_1,hidden_size_2)
        self.ff3 = nn.Linear(hidden_size_2,output_size)
    
    def forward(self,x):
        x1 = self.dropout1(self.prelu(self.ff1(x)))
        x2 = self.dropout2(self.prelu(self.ff2(x1)))
        x3 = self.softmax(self.ff3(x2))
        return x3



#%%