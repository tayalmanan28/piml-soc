import torch
from torch.autograd import grad
from torch.autograd.functional import hessian
# batched jacobian
# y: [..., N], x: [..., M] -> [..., N, M]


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    jac = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[..., i, :] = grad(y_flat, x, torch.ones_like(
            y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, torch.ones_like(
            f), create_graph=True)[0]
        f = grads.sum()

    return grads


def batchHessian(y, x):
    hes = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        hes[..., i, :] = nth_derivative(y_flat, x, 2)

    status = 0
    if torch.any(torch.isnan(hes)):
        status = -1

    return hes, status

