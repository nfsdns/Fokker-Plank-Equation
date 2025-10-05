import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#Version 1
# -------------------------
# Define Problem Examples
# -------------------------
examples = {
    1: {"A": lambda x, u: -u, "B": lambda x, u: u, "u_exact": lambda x, t: x + t, "x0": lambda t: t, "x1": lambda t: t + 1, "f_x": lambda x: x},
    2: {"A": lambda x, u: x * u, "B": lambda x, u: (x**2 / 2.0) * u, "u_exact": lambda x, t: x * torch.exp(t), "x0": lambda t: 0, "x1": lambda t: torch.exp(t), "f_x": lambda x: x},
    3: {"A": lambda x, u: -(x + 1) * u, "B": lambda x, t, u: x**2 * torch.exp(t) * u, "u_exact": lambda x, t: (x + 1) * torch.exp(t), "x0": lambda t: torch.exp(t), "x1": lambda t: 2*torch.exp(t), "f_x": lambda x: x + 1}
}
# -------------------------
# Function to Compute Derivatives
# -------------------------
def derivative(u, inp):
    return torch.autograd.grad(u, inp, grad_outputs=torch.ones_like(u), create_graph=True)[0]

# -------------------------
# Define the Neural Network (Configurable)
# -------------------------
class PINN(nn.Module):
    def __init__(self, activation='tanh'):
        super().__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 10) 
        self.l3 = nn.Linear(10, 1)

        activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'swish': nn.SiLU()}
        self.activation = activations.get(activation, nn.Tanh())

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return x
        # -------------------------
# Training
# -------------------------
def train_pinn(example_id, activation='tanh', lr=0.001, n_epochs=10000, tol=1e-6):
    example = examples[example_id]

    x = torch.linspace(0, 1, 11, requires_grad=True)
    t = torch.linspace(0, 1, 5, requires_grad=True)
    x, t = torch.meshgrid(x, t, indexing='ij')
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)

    model = PINN(activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    prev_loss = float('inf')
    for i in range(n_epochs):
        x_t = torch.cat((x, t), 1)
        u = model(x_t)

        u_t = derivative(u, t)
        u_x = derivative(u, x)
        u_xx = derivative(u_x, x)

        AU = example['A'](x, t,u)
        AU_x = derivative(AU, x)
        BU = example['B'](x, t, u)
        BU_xx = derivative(derivative(BU, x), x)

        loss_residual = ((u_t + AU_x - BU_xx) ** 2).mean()

        t0 = torch.zeros_like(t)
        x_t0 = torch.cat((x, t0), 1)
        loss_initial = ((model(x_t0) - example['f_x'](x)) ** 2).mean()  #TODO:

        x0 = torch.zeros_like(x)
        x0_t = torch.cat((x0, t), 1)
        loss_b1 = ((model(x0_t) - example['x0'](t)) ** 2).mean() #TODO

        x1 = torch.ones_like(x)
        x1_t = torch.cat((x1, t), 1)
        loss_b2 = ((model(x1_t) - example['x1'](t)) ** 2).mean() #TODO

        loss_total = loss_residual + loss_initial + loss_b1 + loss_b2

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        # scheduler.step(loss_total)

        # if abs(prev_loss - loss_total.item()) < tol:
        #     print(f'Early stopping at epoch {i}, Loss = {loss_total.item():.4e}')
        #     break
        # prev_loss = loss_total.item()

        # if i % 1000 == 0:
        #     print(f'Example {example_id} - Epoch {i}: Loss = {loss_total.item():.4e}')

    return model, x, t, example['u_exact'](x, t)
# -------------------------
# Run All Examples and Plot Results
# -------------------------
def plot_results(example_id, x, t, u_pred, u_exact):
    x = x.reshape(50, 10).detach().numpy()
    t = t.reshape(50, 10).detach().numpy()
    u_pred = u_pred.reshape(50, 10).detach().numpy()
    u_exact = u_exact.reshape(50, 10).detach().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    axs[0].plot_surface(x, t, u_pred, cmap='viridis')
    axs[0].set_title(f'Example {example_id} - Predicted')
    axs[1].plot_surface(x, t, u_exact, cmap='viridis')
    axs[1].set_title(f'Example {example_id} - Exact')
    axs[2].plot_surface(x, t, np.abs(u_exact - u_pred), cmap='inferno')
    axs[2].set_title(f'Example {example_id} - Error')
    plt.show()
    def compute_errors(u_exact, u_pred):
        m = u_exact.numel()
        error = u_exact - u_pred
        RMS = torch.sqrt(torch.sum(error ** 2) / m)
        Ne = torch.sqrt(torch.sum(error ** 2) / torch.sum(u_exact ** 2))
        MAE = torch.mean(torch.abs(error))
        return RMS.item(), Ne.item(), MAE.item()
