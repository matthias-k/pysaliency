import math

import torch
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
