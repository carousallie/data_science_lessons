#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot

# sudo pip3 install graphviz
# sudo pip3 install git+https://github.com/szagoruyko/pytorchviz

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, R).cuda())
t = Variable(torch.randn(Batch_size, a_size).cuda(), requires_grad=False)


model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)

model.cuda()
performance_index = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for index in range(500):

    a = model(p)

    loss = performance_index(a, t)
    print(index, loss.data[0])


    model.zero_grad()

    loss.backward()


    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

a = model(p)
g = make_dot(a,params=model.parameters())
g.view()