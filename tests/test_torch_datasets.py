from pathlib import Path

from PIL import Image
import numpy as np
import pytest

from pysaliency import (
    FileStimuli,
    GaussianSaliencyMapModel,
    DigitizeMapModel,
    SaliencyMapModelFromDirectory,
    UniformModel
)
from pysaliency.torch_datasets import ImageDataset, ImageDatasetSampler, FixationMaskTransform, collate_fn
import torch


@pytest.fixture
def stimuli(tmp_path):
    filenames = []
    stimuli_directory = tmp_path / 'stimuli'
    stimuli_directory.mkdir()
    for i in range(50):
        image = Image.fromarray(np.random.randint(0, 255, size=(25, 30, 3), dtype=np.uint8))
        filename = stimuli_directory / 'stimulus_{:04d}.png'.format(i)
        image.save(filename)
        filenames.append(filename)
    return FileStimuli(filenames)


@pytest.fixture
def fixations(stimuli):
    return UniformModel().sample(stimuli, 1000, rst=np.random.RandomState(seed=42))


@pytest.fixture
def saliency_model():
    return GaussianSaliencyMapModel(center_x=0.15, center_y=0.85, width=0.2)


@pytest.fixture
def png_saliency_map_model(tmp_path, stimuli, saliency_model):
    digitized_model = DigitizeMapModel(saliency_model)
    output_path = tmp_path / 'saliency_maps'
    output_path.mkdir()

    for filename, stimulus in zip(stimuli.filenames, stimuli):
        stimulus_name = Path(filename)
        output_filename = output_path / f"{stimulus_name.stem}.png"
        image = Image.fromarray(digitized_model.saliency_map(stimulus).astype(np.uint8))
        image.save(output_filename)

    return SaliencyMapModelFromDirectory(stimuli, str(output_path))


def test_dataset(stimuli, fixations, png_saliency_map_model):
    models_dict = {
        'saliency_map': png_saliency_map_model,
    }

    dataset = ImageDataset(
        stimuli,
        fixations,
        models=models_dict,
        transform=FixationMaskTransform(),
        average='image',
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=4, shuffle=False),
        pin_memory=False,
        num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
        collate_fn=collate_fn,
    )

    count = 0
    for batch in loader:
        count += len(batch['saliency_map'])

    assert count == len(stimuli)
