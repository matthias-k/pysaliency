# Submission

This directory contains an example of how to create a Docker container for submitting a scanpath model to the MIT/Tübingen Saliency Benchmark. You'll need to build a docker or singularity container that offers a json API for requesting model predictions. The benchmark will use `pysaliency.http_models.HTTPScanpathModel` to interact with your model.

## Preparing the submission

1. Create a docker or singularity container that exposes your model as an API compatible with `pysaliency.http_models.HTTPScanpathModel`. There are two different examples contained here:
    - `docker_pysaliency`: A docker container exposing a pysaliency model (which is implemented in `sample_submission.py`). Use this if you already have a pysaliency implementation of your model.
    - `docker_deepgaze`: A docker container exposing the DeepGaze model. It demonstrates how to implement the API for an arbitrary model.

2. Build the Docker container as described in the "Launching the submission container" section.


## Launching the submission container

In this example, we will use the `docker_pysaliency` directory to create a Docker container that exposes a pysaliency model. The container will run a Flask server that listens for HTTP requests and responds with model predictions.

First we have to build the container
```bash
docker build -t sample_pysaliency docker
```

Then we can start it
```bash
docker run --rm -it -p 4000:4000 sample_pysaliency
```
The above command will launch the image as interactive container in the foregroun
and expose the port `4000` to the host machine.
If you prefer to run it in the background, use

```bash
docker run --name sample_pysaliency -dp 4000:4000 sample_pysaliency
```
which will launch a container named `sample_pysaliency`. The container will be running in the background.

To test the model server, run the sample_evaluation script. This script will evaluate the model on the MIT1003 dataset.  Make sure to have the `pysaliency` package installed:
```bash
python ./sample_evaluation.py
```

To delete the background container, run the following command:
```bash
docker stop sample_pysaliency && docker rm sample_pysaliency
```

# TODOs

- [ ] Establish and discuss how arguments can be passed to the model server, e.g. information about the image resolution in dva or other parameters.