import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import Model
from .optpy import minimize
from .saliency_map_models import SaliencyMapModel
from .torch_utils import GaussianFilterNd, Nonlinearity, zero_grad, log_likelihood
from .torch_datasets import ImageDataset, ImageDatasetSampler, FixationMaskTransform


class CenterBias(nn.Module):
    def __init__(self, ys=None, num_values=12):
        super().__init__()
        if ys is None:
            ys = np.linspace(1.1, 0.9, num=num_values, dtype=np.float32)

        self.nonlinearity = Nonlinearity(ys=ys)
        self.alpha = nn.Parameter(torch.tensor(0.5 * np.sqrt(2), dtype=torch.float32), requires_grad=True)

    def forward(self, tensor):
        _, _, height, width = tensor.shape

        x_coords = torch.linspace(-1.0, 1.0, steps=width, device=tensor.device) + 0.000001
        y_coords = torch.linspace(-1.0, 1.0, steps=height, device=tensor.device) + 0.000001  # We cannot have zeros in there because of grad

        x_coords = x_coords[None, None, None, :]
        y_coords = y_coords[None, None, :, None]

        beta_squared = 1.0 - self.alpha**2
        dists = torch.sqrt(
            x_coords**2 / self.alpha**2 + y_coords**2 / beta_squared
        )
        max_dist = torch.sqrt(1.0 / self.alpha**2 + 1.0 / beta_squared)
        dists = dists / max_dist

        centerbias = self.nonlinearity(dists)

        return centerbias


class SaliencyMapProcessing(nn.Module):
    def __init__(self, num_nonlinearity=20, num_centerbias=12, blur_radius=1.0, nonlinearity_target='density', nonlinearity_values='density'):
        super().__init__()

        if nonlinearity_target not in ['density', 'logdensity']:
            raise ValueError("nonlinearity target '{}' not allowed".format(nonlinearity_target))

        if nonlinearity_values not in ['density', 'logdensity']:
            raise ValueError("nonlinearity values '{}' not allowed".format(nonlinearity_values))

        self.blur = GaussianFilterNd(dims=(2, 3), sigma=blur_radius, trainable=True)

        self.nonlinearity_target = nonlinearity_target

        if nonlinearity_target == 'density' and nonlinearity_values == 'logdensity':
            self.nonlinearity = Nonlinearity(value_scale='log')
            with torch.no_grad():
                self.nonlinearity.ys.mul_(8.0)
        elif nonlinearity_target == 'density' and nonlinearity_values == 'logdensity':
            raise ValueError("Invalid combination of nonlinearity target and values")
        elif nonlinearity_target == nonlinearity_values:
            self.nonlinearity = Nonlinearity(value_scale='linear')

        self.centerbias = CenterBias(num_values=num_centerbias)

    def forward(self, tensor):
        tensor = self.blur(tensor)
        tensor = self.nonlinearity(tensor)

        centerbias = self.centerbias(tensor)
        if self.nonlinearity_target == 'density':
            tensor *= centerbias
        elif self.nonlineary_target == 'logdensity':
            tensor += centerbias
        else:
            raise ValueError(self.nonlinearity_target)

        if self.nonlinearity_target == 'density':
            sums = torch.sum(tensor, dim=(2, 3), keepdim=True)
            tensor = tensor / sums
            tensor = torch.log(tensor)
        elif self.nonlineary_target == 'logdensity':
            logsums = torch.logsumexp(tensor, dim=(2, 3), keepdim=True)
            tensor = tensor - logsums
        else:
            raise ValueError(self.nonlinearity_target)

        return tensor


class NormalizedSaliencyMapModel(SaliencyMapModel):
    def __init__(self, parent_model, saliency_min=None, saliency_max=None, caching=False, **kwargs):
        super().__init__(caching=caching, **kwargs)
        self.parent_model = parent_model
        self.saliency_min = saliency_min
        self.saliency_max = saliency_max

    def _saliency_map(self, stimulus):
        saliency_map = self.parent_model.saliency_map(stimulus)

        if self.saliency_min is not None:
            minimum = self.saliency_min
        else:
            minimum = saliency_map.min()

        if self.saliency_max is not None:
            maximum = self.saliency_max
        else:
            maximum = saliency_map.max()

        normalized_saliency_map = (saliency_map - minimum) / (maximum - minimum)
        return np.minimum(1, np.maximum(0, normalized_saliency_map))


def run_dataset(model, dataset, device, verbose=True):
    """ processes dataset and accumulate gradients """
    model.train()
    losses = []
    batch_weights = []

    pbar = tqdm(dataset, disable=not verbose)
    for batch in pbar:

        fixation_mask = batch['fixation_mask'].to(device)
        weights = batch['weight'].to(device)

        saliency_maps = []
        while 'prediction_{:04d}'.format(len(saliency_maps)) in batch:
            saliency_maps.append(batch['prediction_{:04d}'.format(len(saliency_maps))].to(device))

        this_loss = 0
        for saliency_map in saliency_maps:
            log_density = model(saliency_map[:, None, :, :])[:, 0, :, :]

            loss = -log_likelihood(log_density, fixation_mask, weights=weights)
            this_loss += loss / len(saliency_maps)

        losses.append(this_loss.detach().cpu().numpy())

        batch_weights.append(weights.detach().cpu().numpy().sum())

        pbar.set_description('{:.05f}'.format(np.average(losses, weights=batch_weights)))

        (weights.detach().sum().detach() * loss).backward()

    total_weight = np.sum(batch_weights)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.div_(total_weight)

    return np.average(losses, weights=batch_weights)


def optimize_saliency_map_conversion(
        model, stimuli, fixations,
        nonlinearity_target='density',
        nonlinearity_values='logdensity',
        saliency_min=None,
        saliency_max=None,
        optimize=None,
        average='image',
        verbose=0,
        method='SLSQP',
        num_nonlinearity=20,
        num_centerbias=12,
        blur_radius=20.0,
        batch_size=8,
        tol=None,
        maxiter=1000,
        minimize_options=None,
        return_optimization_result=False):

    targets = [([model], stimuli, fixations)]

    if saliency_min is None or saliency_max is None:
        smax = -np.inf
        smin = np.inf
        for s in tqdm(stimuli):
            smap = model.saliency_map(s)
            smax = np.max([smax, smap.max()])
            smin = np.min([smin, smap.min()])

        if saliency_min is None:
            saliency_min = smin
        if saliency_max is None:
            saliency_max = smax

    saliency_map_processing, optimization_result = _optimize_saliency_map_conversion_over_multiple_models_and_datasets(
        targets,
        nonlinearity_target=nonlinearity_target,
        nonlinearity_values=nonlinearity_values,
        saliency_min=saliency_min,
        saliency_max=saliency_max,
        optimize=optimize,
        average=average,
        verbose=verbose,
        method=method,
        num_nonlinearity=num_nonlinearity,
        num_centerbias=num_centerbias,
        blur_radius=blur_radius,
        batch_size=batch_size,
        tol=tol,
        maxiter=maxiter,
        minimize_options=minimize_options)

    return_model = SaliencyMapProcessingModel(
        model,
        nonlinearity_target=nonlinearity_target,
        nonlinearity_values=nonlinearity_values,
        saliency_min=saliency_min,
        saliency_max=saliency_max,
        num_nonlinearity=num_nonlinearity,
        num_centerbias=num_centerbias,
        blur_radius=blur_radius,
        saliency_map_processing=saliency_map_processing
    )

    if return_optimization_result:
        return return_model, optimization_result

    return return_model


def _optimize_saliency_map_conversion_over_multiple_models_and_datasets(
        list_of_targets,
        nonlinearity_target='density',
        nonlinearity_values='density',
        saliency_min=None,
        saliency_max=None,
        optimize=None,
        average='image',
        verbose=0,
        method='SLSQP',
        num_nonlinearity=20,
        num_centerbias=12,
        blur_radius=20.0,
        batch_size=8,
        tol=None,
        maxiter=1000,
        minimize_options=None):

    if len(list_of_targets) != 1:
        raise NotImplementedError()

    models, stimuli, fixations = list_of_targets[0]

    models_dict = {
        'prediction_{:04d}'.format(i): NormalizedSaliencyMapModel(
            model,
            saliency_min=saliency_min,
            saliency_max=saliency_max,
        ) for i, model in enumerate(models)
    }

    dataset = ImageDataset(
        stimuli,
        fixations,
        models=models_dict,
        transform=FixationMaskTransform(),
        average=average,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size, shuffle=False),
        pin_memory=False,
        num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
    )

    saliency_map_processing = SaliencyMapProcessing(
        nonlinearity_values=nonlinearity_values,
        nonlinearity_target=nonlinearity_target,
        num_nonlinearity=num_nonlinearity,
        num_centerbias=num_centerbias,
        blur_radius=blur_radius,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device", device)
    saliency_map_processing.to(device)

    optimization_result = _optimize_saliency_map_processing(
        saliency_map_processing,
        loader,
        verbose=verbose,
        optimize=optimize,
        method=method,
        tol=tol,
        maxiter=maxiter,
        minimize_options=minimize_options,
    )

    return saliency_map_processing, optimization_result


def _optimize_saliency_map_processing(
        saliency_map_processing,
        data_loader,
        optimize=None,
        verbose=0,
        method='SLSQP',
        tol=None,
        maxiter=1000,
        minimize_options=None):

    if optimize is None:
        optimize = ['blur_radius', 'nonlinearity', 'centerbias', 'alpha']

    full_param_dict = {
        'blur_radius': saliency_map_processing.blur.sigma,
        'nonlinearity': saliency_map_processing.nonlinearity.ys,
        'centerbias': saliency_map_processing.centerbias.nonlinearity.ys,
        'alpha': saliency_map_processing.centerbias.alpha,
    }

    nonlinearity_target = saliency_map_processing.nonlinearity_target
    nonlinearity_values = saliency_map_processing.nonlinearity.value_scale
    num_nonlinearity = len(saliency_map_processing.nonlinearity.ys)

    initial_params = {key: value.detach().cpu().numpy() for key, value in full_param_dict.items()}

    def func(blur_radius, nonlinearity, centerbias, alpha, optimize=None):
        if verbose > 5:
            print('blur_radius: ', blur_radius)
            print('nonlinearity:', nonlinearity)
            print('centerbias:  ', centerbias)
            print('alpha:       ', alpha)

        def set_param(param, value):
            param.copy_(torch.tensor(value))

        with torch.no_grad():
            set_param(saliency_map_processing.blur.sigma, blur_radius)
            set_param(saliency_map_processing.nonlinearity.ys, nonlinearity)
            set_param(saliency_map_processing.centerbias.nonlinearity.ys, centerbias)
            set_param(saliency_map_processing.centerbias.alpha, alpha)

        zero_grad(saliency_map_processing)

        loss = run_dataset(
            saliency_map_processing,
            data_loader,
            saliency_map_processing.nonlinearity.ys.device,
            verbose=verbose > 4
        )

        gradients = [
            full_param_dict[param_name].grad.detach().cpu().numpy() for param_name in optimize
        ]

        return loss, tuple(gradients)

    bounds = {
        'alpha': [(1e-4, 1.0 - 1e-4)],
        'blur_radius': [(0.0, 1e3)]
    }
    constraints = []

    if 'nonlinearity' in optimize:
        # mononotic nonlinearity
        for i in range(1, num_nonlinearity):
            constraints.append({
                'type': 'ineq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha, i=i: nonlinearity[i] - nonlinearity[i - 1]
            })

    if nonlinearity_target == 'density':
        bounds['centerbias'] = [(1e-6, 1000) for i in range(len(saliency_map_processing.centerbias.nonlinearity.ys))]
        if 'centerbias' in optimize:
            constraints.append({
                'type': 'eq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias.sum() - initial_params['centerbias'].sum()
            })

    elif nonlinearity_target == 'logdensity':
        bounds['centerbias'] = [(None, None) for i in range(len(saliency_map_processing.centerbias.nonlinearity.ys))]
        if 'centerbias' in optimize:
            constraints.append({
                'type': 'eq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: centerbias[0]
            })
    else:
        raise ValueError(nonlinearity_target)

    if (nonlinearity_target == 'density') and (nonlinearity_values == 'linear'):
        bounds['nonlinearity'] = [(1e-6, 1e7) for i in range(num_nonlinearity)]
        if 'nonlinearity' in optimize:
            constraints.append({
                'type': 'eq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity.sum() - initial_params['nonlinearity'].sum()
            })

    elif (nonlinearity_target == 'density') and (nonlinearity_values == 'log'):
        bounds['nonlinearity'] = [(-100, 100) for i in range(num_nonlinearity)]
        if 'nonlinearity' in optimize:
            constraints.append({
                'type': 'eq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: np.exp(nonlinearity).sum() - np.exp(initial_params['nonlinearity']).sum()
            })

    elif (nonlinearity_target == 'logdensity') and (nonlinearity_values == 'linear'):
        bounds['nonlinearity'] = [(None, None) for i in range(num_nonlinearity)]
        if 'nonlinearity' in optimize:
            constraints.append({
                'type': 'eq',
                'fun': lambda blur_radius, nonlinearity, centerbias, alpha: nonlinearity[0]
            })

    else:
        raise ValueError(nonlinearity_target, nonlinearity_values)

    if method == 'SLSQP':
        options = {'iprint': 2, 'disp': 2, 'maxiter': maxiter,
                   'eps': 1e-9
                   }
        tol = tol or 1e-9
    elif method == 'IPOPT':
        tol = tol or 1e-7
        options = {'disp': 5, 'maxiter': maxiter,
                   'tol': tol
                   }
    else:
        options = {'maxiter': maxiter}

    if minimize_options:
        options.update(minimize_options)

    x0 = initial_params

    res = minimize(func, x0, jac=True, constraints=constraints, bounds=bounds, method=method, tol=tol, options=options, optimize=optimize)
    return res


class SaliencyMapProcessingModel(Model):
    def __init__(
            self,
            saliency_map_model,
            nonlinearity_values='logdensity',
            nonlinearity_target='density',
            saliency_min=None,
            saliency_max=None,
            num_nonlinearity=20,
            num_centerbias=12,
            blur_radius=20.0,
            saliency_map_processing=None,
            device=None,
            **kwargs):

        super().__init__(**kwargs)
        self.saliency_map_model = saliency_map_model

        self.normalized_saliency_map_model = NormalizedSaliencyMapModel(
            saliency_map_model,
            saliency_min=saliency_min,
            saliency_max=saliency_max,
        )

        if saliency_map_processing is None:
            saliency_map_processing = SaliencyMapProcessing(
                nonlinearity_values=nonlinearity_values,
                nonlinearity_target=nonlinearity_target,
                num_nonlinearity=num_nonlinearity,
                num_centerbias=num_centerbias,
                blur_radius=blur_radius,
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device", device)
        self.device = device

        saliency_map_processing.to(self.device)
        self.saliency_map_processing = saliency_map_processing

    def _log_density(self, stimulus):
        saliency_map = self.normalized_saliency_map_model.saliency_map(stimulus)
        saliency_map_tensor = torch.tensor(saliency_map[np.newaxis, np.newaxis, :, :]).to(self.device)
        return self.saliency_map_processing.forward(saliency_map_tensor).detach().cpu().numpy()[0, 0, :, :]
