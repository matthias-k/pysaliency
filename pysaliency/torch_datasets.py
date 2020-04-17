import random

from boltons.iterutils import chunked
import numpy as np
import torch
from tqdm import tqdm

from .models import Model
from .saliency_map_models import SaliencyMapModel


def ensure_color_image(image):
    if len(image.shape) == 2:
        return np.dstack([image, image, image])
    return image


def x_y_to_sparse_indices(xs, ys):
    # Converts list of x and y coordinates into indices and values for sparse mask
    x_inds = []
    y_inds = []
    values = []
    pair_inds = {}

    for x, y in zip(xs, ys):
        key = (x, y)
        if key not in pair_inds:
            x_inds.append(x)
            y_inds.append(y)
            pair_inds[key] = len(x_inds) - 1
            values.append(1)
        else:
            values[pair_inds[key]] += 1

    return np.array([y_inds, x_inds]), values


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, stimuli, fixations, models=None, transform=None, cached=True, average='fixation'):
        self.stimuli = stimuli
        self.fixations = fixations

        if models is None:
            models = {}

        self.models = models
        self.transform = transform
        self.average = average

        self.cached = cached
        if cached:
            self._cache = {}
            print("Populating fixations cache")
            self._xs_cache = {}
            self._ys_cache = {}

            for x, y, n in zip(self.fixations.x_int, self.fixations.y_int, tqdm(self.fixations.n)):
                self._xs_cache.setdefault(n, []).append(x)
                self._ys_cache.setdefault(n, []).append(y)

            for key in list(self._xs_cache):
                self._xs_cache[key] = np.array(self._xs_cache[key], dtype=np.long)
            for key in list(self._ys_cache):
                self._ys_cache[key] = np.array(self._ys_cache[key], dtype=np.long)

    def get_shapes(self):
        return list(self.stimuli.sizes)

    def __getitem__(self, key):
        if not self.cached or key not in self._cache:
            image = np.array(self.stimuli.stimuli[key])

            predictions = {}
            for model_name, model in self.models.items():
                if isinstance(model, Model):
                    prediction = model.log_density(image)
                elif isinstance(model, SaliencyMapModel):
                    prediction = model.saliency_map(image)
                predictions[model_name] = prediction

            image = ensure_color_image(image).astype(np.float32)
            image = image.transpose(2, 0, 1)

            if self.cached:
                xs = self._xs_cache.pop(key)
                ys = self._ys_cache.pop(key)
            else:
                inds = self.fixations.n == key
                xs = np.array(self.fixations.x_int[inds], dtype=np.long)
                ys = np.array(self.fixations.y_int[inds], dtype=np.long)
            data = {
                "image": image,
                "x": xs,
                "y": ys,
            }

            for prediction_name, prediction in predictions.items():
                data[prediction_name] = prediction

            if self.average == 'image':
                data['weight'] = 1.0
            else:
                data['weight'] = float(len(xs))

            if self.cached:
                self._cache[key] = data
        else:
            data = self._cache[key]

        if self.transform is not None:
            return self.transform(dict(data))

        return data

    def __len__(self):
        return len(self.stimuli)


class FixationMaskTransform(object):
    def __call__(self, item):
        shape = torch.Size([item['image'].shape[1], item['image'].shape[2]])
        x = item.pop('x')
        y = item.pop('y')

        # inds, values = x_y_to_sparse_indices(x, y)
        inds = np.array([y, x])
        values = np.ones(len(y), dtype=np.int)

        mask = torch.sparse.IntTensor(torch.tensor(inds), torch.tensor(values), shape)
        mask = mask.coalesce()

        item['fixation_mask'] = mask

        return item


class ImageDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=1, ratio_used=1.0, shuffle=True):
        self.ratio_used = ratio_used
        self.shuffle = shuffle

        shapes = data_source.get_shapes()
        unique_shapes = sorted(set(shapes))

        shape_indices = [[] for shape in unique_shapes]

        for k, shape in enumerate(shapes):
            shape_indices[unique_shapes.index(shape)].append(k)

        if self.shuffle:
            for indices in shape_indices:
                random.shuffle(indices)

        self.batches = sum([chunked(indices, size=batch_size) for indices in shape_indices], [])

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.batches))
        else:
            indices = range(len(self.batches))

        if self.ratio_used < 1.0:
            indices = indices[:int(self.ratio_used * len(indices))]

        return iter(self.batches[i] for i in indices)

    def __len__(self):
        return int(self.ratio_used * len(self.batches))
