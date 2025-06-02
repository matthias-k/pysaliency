# Submission
TODO: Add a description of the submission process here.



## Launching the submission container

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

To test the model server, run the sample_evaluation script (Make sure to have the `pysaliency` package installed):
```bash
python ./sample_evaluation.py
```

To delete the background container, run the following command:
```bash
docker stop sample_pysaliency && docker rm sample_pysaliency
```