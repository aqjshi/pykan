from kan import KAN, LBFGS
import torch
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import pandas as pd

'''
params
'''
S = 2
T = 1
t = 0
tau = T-t
K = 5
sigma = 0.13
r = 0.03
x = np.log(S)
X_max = 2*K

dim = 2
np_i = 50  # Number of interior points (along each dimension)
np_b = 50  # Number of boundary points (along each dimension)
ranges_X = [0, X_max]
ranges_tau = [0, T]


'''
FUNCTION DEFINITIONS
'''
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return price

def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)
    
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)

def helper(X, Y):
    return torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1, 0)


'''
solu
'''

# Define the solution functions
d1 = lambda x: (x[:, [0]] - torch.log(torch.tensor(K)) + (r + 0.5 * sigma**2) * x[:, [1]]) / (sigma * torch.sqrt(x[:, [1]]))
d2 = lambda x: d1(x) - sigma * torch.sqrt(x[:, [1]])

sol_fun = lambda x: torch.exp(x[:, [0]]) * torch.tensor(norm.cdf(d1(x).numpy())) - K * torch.exp(-r * x[:, [1]]) * torch.tensor(norm.cdf(d2(x).numpy()))

source_fun = lambda x: 0



'''
data prep
'''
# Define grid and sampling
device = "cpu"
sampling_mode = 'random'  # Options: 'random' or 'mesh'
x_mesh = torch.linspace(ranges_X[0], ranges_X[1], steps=np_i)
y_mesh = torch.linspace(ranges_tau[0], ranges_tau[1], steps=np_i)
X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")

if sampling_mode == 'mesh':
    x_i = helper(X, Y)
else:
    x_i = torch.rand((np_i**2, 1), device=device) * (ranges_X[1] - ranges_X[0]) + ranges_X[0]
    y_i = torch.rand((np_i**2, 1), device=device) * (ranges_tau[1] - ranges_tau[0]) + ranges_tau[0]
    x_i = torch.cat([x_i, y_i], dim=1)

# Split into 80% train and 20% test
num_train = int(0.8 * x_i.shape[0])
indices = torch.randperm(x_i.shape[0])

train_indices = indices[:num_train]
test_indices = indices[num_train:]

x_i_train = x_i[train_indices]
x_i_test = x_i[test_indices]

# Prepare boundary data
xb1 = helper(X[0], Y[1])
xb2 = helper(X[-1], Y[0])
xb3 = helper(X[:, 0], Y[:, 1])
x_b = torch.cat([xb1, xb2, xb3], dim=0)

'''
model def
'''

# Define the model
model = KAN(width=[2, 2, 1], grid=5, k=3, grid_eps=1.0, noise_scale=0.25)
def train(steps=50, alpha=0.01, log=1):
    optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='Training')  # Progress bar for training
    for step in pbar: 
        def closure():
            global pde_loss, bc_loss
            optimizer.zero_grad()
            
            # Compute interior loss using training data
            sol_train = sol_fun(x_i_train)
            sol_D1_fun = lambda x: batch_jacobian(model, x, create_graph=True)[:, 0, :]
            sol_D1 = sol_D1_fun(x_i_train)
            sol_D2 = batch_jacobian(sol_D1_fun, x_i_train, create_graph=True)[:, :, :]
            lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
            source = source_fun(x_i_train)
            pde_loss = torch.mean((lap - source)**2)

            # Compute boundary loss
            bc_true = sol_fun(x_b)
            bc_pred = model(x_b)
            bc_loss = torch.mean((bc_pred - bc_true)**2)
            
            # Total loss is a weighted sum of PDE loss and boundary loss
            loss = alpha * pde_loss + bc_loss
            loss.backward()  # Backpropagation

            return loss

        if step % 5 == 0 and step < 50:
            model.update_grid_from_samples(x_i_train)  # Update grid from samples during training
        
        optimizer.step(closure)  # Perform a single optimization step

        # Print the gradients for each parameter at the end of the step
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Step {step}: Gradient for {name} - {param.grad}")
            else:
                print(f"Step {step}: No gradient for {name}")

        # Evaluate the model on the training data
        sol_train = sol_fun(x_i_train)
        l2_train = torch.mean((model(x_i_train) - sol_train)**2)

        # Optionally, evaluate the model on the test data
        sol_test = sol_fun(x_i_test)
        l2_test = torch.mean((model(x_i_test) - sol_test)**2)

        if step % log == 0:
            pbar.set_description("pde loss: %.2e | bc loss: %.2e | l2 train: %.2e | l2 test: %.2e" % 
                                 (pde_loss.cpu().detach().numpy(), 
                                  bc_loss.cpu().detach().numpy(), 
                                  l2_train.detach().numpy(),
                                  l2_test.detach().numpy()))


train()

'''
eval 
'''
# Evaluation and plotting
model.plot(beta=10)





