import math

from boltons.iterutils import windowed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_filter_1d(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0):
    sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)

    if kernel_size is not None:
        kernel_size = torch.as_tensor(kernel_size, device=tensor.device, dtype=torch.int64)
    else:
        kernel_size = torch.as_tensor(2 * torch.ceil(truncate * sigma) + 1, device=tensor.device, dtype=torch.int64)

    kernel_size = kernel_size.detach()

    kernel_size_int = kernel_size.detach().cpu().numpy()

    mean = (torch.as_tensor(kernel_size, dtype=tensor.dtype) - 1) / 2

    grid = torch.arange(kernel_size, device=tensor.device) - mean

    # reshape the grid so that it can be used as a kernel for F.conv1d
    kernel_shape = [1] * len(tensor.shape)
    kernel_shape[dim] = kernel_size_int
    grid = grid.view(kernel_shape)

    grid = grid.detach()

    padding = [0] * (2 * len(tensor.shape))
    padding[dim * 2 + 1] = math.ceil((kernel_size_int - 1) / 2)
    padding[dim * 2] = math.ceil((kernel_size_int - 1) / 2)
    padding = tuple(reversed(padding))

    if padding_mode in ['replicate']:
        # replication padding has some strange constraints...
        assert len(tensor.shape) - dim <= 2
        padding = padding[:(len(tensor.shape) - 2) * 2]

    tensor_ = F.pad(tensor, padding, padding_mode, padding_value)

    # create gaussian kernel from grid using current sigma
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # convolve input with gaussian kernel
    return F.conv1d(tensor_, kernel)


def gaussian_filter(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0):
    if isinstance(dim, int):
        dim = [dim]

    for k in dim:
        tensor = gaussian_filter_1d(
            tensor,
            dim=k,
            sigma=sigma,
            truncate=truncate,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )

    return tensor


class GaussianFilterNd(nn.Module):
    """A differentiable gaussian filter"""

    def __init__(self, dims, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0,
                 trainable=False):
        """Creates a 1d gaussian filter

        Args:
            dims ([int]): the dimensions to which the gaussian filter is applied. Negative values won't work
            sigma (float): standard deviation of the gaussian filter (blur size)
            truncate (float, optional): truncate the filter at this many standard deviations (default: 4.0).
                This has no effect if the `kernel_size` is explicitely set
            kernel_size (int): size of the gaussian kernel convolved with the input
            padding_mode (string, optional): Padding mode implemented by `torch.nn.functional.pad`.
            padding_value (string, optional): Value used for constant padding.
        """
        # IDEA determine input_dims dynamically for every input
        super(GaussianFilterNd, self).__init__()

        self.dims = dims
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=trainable)  # default: no optimization
        self.truncate = truncate
        self.kernel_size = kernel_size

        # setup padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, tensor):
        """Applies the gaussian filter to the given tensor"""
        for dim in self.dims:
            tensor = gaussian_filter_1d(
                tensor,
                dim=dim,
                sigma=self.sigma,
                truncate=self.truncate,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                padding_value=self.padding_value,
            )

        return tensor


def nonlinearity(tensor, xs, ys):
    assert len(xs) == len(ys)

    output = torch.ones_like(tensor, device=tensor.device) * ys[0]

    for (x1, x2), (y1, y2) in zip(windowed(xs, 2), windowed(ys, 2)):
        output += (y2 - y1) / (x2 - x1) * (torch.clamp(tensor, x1, x2) - x1)

    return output


class Nonlinearity(nn.Module):
    def __init__(self, xs=None, ys=None, num_values=20, value_scale='linear'):
        super().__init__()

        if value_scale not in ['linear', 'log']:
            raise ValueError(value_scale)

        self.value_scale = value_scale

        if ys is None:
            if self.value_scale == 'linear':
                ys = np.linspace(0, 1, num_values)
            elif self.value_scale == 'log':
                ys = np.linspace(-1, 0, num_values)
            else:
                raise ValueError(self.value_scale)

        self.ys = nn.Parameter(data=torch.tensor(ys), requires_grad=True)

        if xs is None:
            xs = np.linspace(0, 1, len(ys))
        self.xs = nn.Parameter(torch.tensor(xs), requires_grad=False)

    def forward(self, tensor):
        ys = self.ys
        if self.value_scale == 'log':
            ys = torch.exp(ys)
        return nonlinearity(tensor, self.xs, ys)


def zero_grad(model):
    """ set gradient of model to zero without having to use an optimizer object """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def log_likelihood(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    dense_mask = fixation_mask.to_dense()
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)
