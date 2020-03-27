import numpy as np
from scipy.ndimage import gaussian_filter as sp_gaussian_filter
from scipy.special import logsumexp
import torch
from torch.optim.optimizer import required
import torch.nn as nn
from tqdm import tqdm

from .models import sample_from_logdensity
from .torch_utils import gaussian_filter


def sample_batch_fixations(log_density, fixations_per_image, batch_size, rst=None):
    xs, ys = sample_from_logdensity(log_density, fixations_per_image * batch_size, rst=rst)
    ns = np.repeat(np.arange(batch_size, dtype=int), repeats=fixations_per_image)

    return xs, ys, ns


class DistributionSGD(torch.optim.Optimizer):
    """Extension of SGD that constraints the parameters to be nonegative and with fixed sum
    (e.g., a probability distribution)"""

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(DistributionSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                learning_rate = group['lr']

                #
                constraint_grad = torch.ones_like(d_p)
                constraint_grad_norm = torch.sum(torch.pow(constraint_grad, 2))
                normed_constraint_grad = constraint_grad / constraint_grad_norm

                # first step: make sure we are not running into negative values
                max_allowed_grad = p / learning_rate
                projected_grad1 = torch.min(d_p, max_allowed_grad)

                # second step: Make sure that the gradient does not walk
                # out of the constraint
                projected_grad2 = projected_grad1 - torch.sum(projected_grad1 * constraint_grad) * normed_constraint_grad

                p.add_(projected_grad2, alpha=-group['lr'])

        return loss


def build_fixation_maps(Ns, Ys, Xs, batch_size, height, width, dtype=torch.float32):
    indices = torch.stack((Ns, Ys, Xs), axis=1).T
    src = torch.ones(indices.shape[1], dtype=dtype, device=indices.device)
    fixation_maps = torch.sparse_coo_tensor(indices, src, size=(batch_size, height, width)).to_dense()

    return fixation_maps


def torch_similarity(saliency_map, empirical_saliency_maps):
    normalized_empirical_saliency_maps = empirical_saliency_maps / torch.sum(empirical_saliency_maps, dim=[1, 2], keepdim=True)
    normalized_saliency_map = saliency_map / torch.sum(saliency_map)
    minimums = torch.min(normalized_empirical_saliency_maps, normalized_saliency_map[None, :, :])

    similarities = torch.sum(minimums, dim=[1, 2])

    return similarities


def compute_similarity(saliency_map, ns, ys, xs, batch_size, kernel_size, truncate_gaussian, dtype=torch.float32):
    height, width = saliency_map.shape
    fixation_maps = build_fixation_maps(ns, ys, xs, batch_size, height, width, dtype=dtype)

    empirical_saliency_maps = gaussian_filter(
        fixation_maps[:, None, :, :],
        dim=[2, 3],
        sigma=kernel_size,
        truncate=truncate_gaussian,
        padding_mode='constant',
        padding_value=0.0,
    )[:, 0, :, :]

    similarities = torch_similarity(saliency_map, empirical_saliency_maps)

    return similarities


class Similarities(nn.Module):
    def __init__(self, initial_saliency_map, kernel_size, truncate_gaussian=3, dtype=torch.float32):
        super().__init__()
        self.saliency_map = nn.Parameter(torch.tensor(initial_saliency_map, dtype=dtype), requires_grad=True)

        self.kernel_size = kernel_size
        self.truncate_gaussian = truncate_gaussian
        self.dtype = dtype

    def forward(self, ns, ys, xs, batch_size):
        similarities = compute_similarity(
            self.saliency_map,
            ns, ys, xs,
            batch_size,
            self.kernel_size,
            self.truncate_gaussian,
            dtype=self.dtype,
        )

        return similarities


def _eval_metric(log_density, test_samples, fn, seed=42, fixation_count=120, batch_size=50, verbose=True):
    values = []
    weights = []
    count = 0

    rst = np.random.RandomState(seed=seed)

    with tqdm(total=test_samples, leave=False, disable=not verbose) as t:
        while count < test_samples:
            this_count = min(batch_size, test_samples - count)
            xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=rst)

            values.append(fn(ns, ys, xs, this_count))
            weights.append(this_count)
            count += this_count
            t.update(this_count)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    return np.average(values, weights=weights)


def maximize_expected_sim(log_density, kernel_size,
                          train_samples_per_epoch, val_samples,
                          train_seed=43, val_seed=42,
                          fixation_count=100, batch_size=50,
                          max_batch_size=None,
                          verbose=True, session_config=None,
                          initial_learning_rate=1e-7,
                          backlook=1, min_iter=0, max_iter=1000,
                          truncate_gaussian=3,
                          learning_rate_decay_samples=None,
                          initial_saliency_map=None,
                          learning_rate_decay_scheme=None,
                          learning_rate_decay_ratio=0.333333333,
                          minimum_learning_rate=1e-11):
    """
       max_batch_size: maximum possible batch size to be used in validation
       learning rate decay samples: how often to decay the learning rate (using 1/k)

       learning_rate_decay_scheme: how to decay the learning rate:
           - None, "1/k": 1/k scheme
           - "validation_loss": if validation loss not better for last backlook
           steps

        learning_rate_decay_ratio: how much to decay learning rate if `learning_rate_decay_scheme` == 'validation_loss'
        minimum_learning_rate: stop optimization if learning rate would drop below this rate if using validation loss decay scheme

    """

    if max_batch_size is None:
        max_batch_size = batch_size

    if learning_rate_decay_scheme is None:
        learning_rate_decay_scheme = '1/k'

    if learning_rate_decay_samples is None:
        learning_rate_decay_samples = train_samples_per_epoch

    log_density_sum = logsumexp(log_density)
    if not -0.001 < log_density_sum < 0.001:
        raise ValueError("Log density not normalized! LogSumExp={}".format(log_density_sum))

    if initial_saliency_map is None:
        initial_value = sp_gaussian_filter(np.exp(log_density), kernel_size, mode='constant')
    else:
        initial_value = initial_saliency_map

    if initial_value.min() < 0:
        initial_value -= initial_value.min()

    initial_value /= initial_value.sum()

    dtype = torch.float32

    model = Similarities(
        initial_saliency_map=initial_value,
        kernel_size=kernel_size,
        truncate_gaussian=truncate_gaussian,
        dtype=dtype
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    model.to(device)

    optimizer = DistributionSGD(model.parameters(), lr=initial_learning_rate)

    if learning_rate_decay_scheme == '1/k':
        def lr_lambda(epoch):
            return 1.0 / (max(epoch, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
        )
    elif learning_rate_decay_scheme == 'validation_loss':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=learning_rate_decay_ratio,
        )
    else:
        raise ValueError(learning_rate_decay_scheme)

    height, width = log_density.shape

    def _val_loss(ns, ys, xs, batch_size):
        model.eval()
        Ns = torch.tensor(ns).to(device)
        Ys = torch.tensor(ys).to(device)
        Xs = torch.tensor(xs).to(device)
        batch_size = torch.tensor(batch_size).to(device)

        ret = -torch.mean(model(Ns, Ys, Xs, batch_size)).detach().cpu().numpy()
        return ret

    def val_loss():
        return _eval_metric(log_density, val_samples, _val_loss, seed=val_seed,
                            fixation_count=fixation_count, batch_size=max_batch_size, verbose=False)

    total_samples = 0
    decay_step = 0

    val_scores = [val_loss()]
    learning_rate_relevant_scores = list(val_scores)
    train_rst = np.random.RandomState(seed=train_seed)

    with tqdm(disable=not verbose) as outer_t:

        def general_termination_condition():
            return len(val_scores) - 1 >= max_iter

        def termination_1overk():
            return not (np.argmin(val_scores) >= len(val_scores) - backlook)

        def termination_validation():
            return optimizer.state_dict()['param_groups'][0]['lr'] < minimum_learning_rate

        def termination_condition():
            if len(val_scores) < min_iter:
                return False
            cond = general_termination_condition()
            if learning_rate_decay_scheme == '1/k':
                cond = cond or termination_1overk()
            elif learning_rate_decay_scheme == 'validation_loss':
                cond = cond or termination_validation()

            return cond

        while not termination_condition():
            count = 0
            with tqdm(total=train_samples_per_epoch, leave=False, disable=True) as t:
                while count < train_samples_per_epoch:
                    model.train()
                    optimizer.zero_grad()
                    this_count = min(batch_size, train_samples_per_epoch - count)

                    xs, ys, ns = sample_batch_fixations(log_density, fixations_per_image=fixation_count, batch_size=this_count, rst=train_rst)

                    Ns = torch.tensor(ns).to(device)
                    Ys = torch.tensor(ys).to(device)
                    Xs = torch.tensor(xs).to(device)
                    batch_size = torch.tensor(batch_size).to(device)

                    loss = -torch.mean(model(Ns, Ys, Xs, batch_size))
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        if torch.sum(model.saliency_map < 0):
                            model.saliency_map.mul_(model.saliency_map >= 0)
                        model.saliency_map.div_(torch.sum(model.saliency_map))

                    count += this_count
                    total_samples += this_count

                    if learning_rate_decay_scheme == '1/k':
                        if total_samples >= (decay_step + 1) * learning_rate_decay_samples:
                            decay_step += 1
                            scheduler.step()

                    t.update(this_count)
            val_scores.append(val_loss())
            learning_rate_relevant_scores.append(val_scores[-1])

            if learning_rate_decay_scheme == 'validation_loss' and np.argmin(learning_rate_relevant_scores) < len(learning_rate_relevant_scores) - backlook:
                scheduler.step()
                learning_rate_relevant_scores = [learning_rate_relevant_scores[-1]]

            score1, score2 = val_scores[-2:]
            last_min = len(val_scores) - np.argmin(val_scores) - 1
            outer_t.set_description('{:.05f}->{:.05f}, diff {:.02e}, best val {} steps ago, lr {:.02e}'.format(val_scores[0], score2, score2 - score1, last_min, optimizer.state_dict()['param_groups'][0]['lr']))
            outer_t.update(1)

    return model.saliency_map.detach().cpu().numpy(), val_scores[-1]
