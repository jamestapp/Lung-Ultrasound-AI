# dl-training-template
This repo provides an example of how to create model code suitable for use on our Kubernetes cluster.

## Dependencies

Running deploy.py requires that the Kubernetes python client is installed for you interpreter:

```pip install kubernetes```

train.py is designed to run in the Docker container and so all of its dependencies are handled when building that.

## Code Description

* deploy
  * image_build
    * **dockerfile** - This is a dockerfie that describes how to build a container that contains all the right dependencies to be run the model training code. That container is then used to run the code in, either locally or on the cluster.
    * **requirements.txt** - This specifies pip packages that are installed in the container during the docker build process. If you change this file then you should rebuild the contaner in order for your changes to take effect.
  * **docker-compose.yml** - The docker-compose.yml file can be used to create a remote interpreter to develop against, and to run code locally. It is also used internally in deploy.py to build the container that is shipped to the cluster. It expects that an environment variable PROJECT_NAME is set and this is used to name the image that is built. The volume mapping can be changed to point to data on your local machine for when running locally. No further edits to this file should typically be required.
  * **deploy.py** - This file submits training jobs to the cluster. You will need to be connected to VPN for it to work. It expects the following arguments (Any other arguments that are supplied are passed along to the entrypoint script):
    * --mode - Whether to submit to the cluster "internal_cluster" ot to run locally "local_machine"
    * --entrypoint - The name of the script to be submitted to the cluster. This will generally be ../source/train.py
    * --project_name - The name of the project used for logging etc
    * (--build_only) - (optional) If this flag is included then the script only builds the docker image, without running the script (useful prior to IDE setup)
  * **templates.py** - This issued internally by deploy.py and does not require changes.
* source - Everything in the source directory will be shipped to the cluster to be run
  * training_lists - Store training lists in this directory
  * **dataloader.py** - Contains dataloader code
  * **model.py** - Contains code that describes the model
  * **train.py** - Contains code that initates model training and logging. This code must be carefully structured in order that it can run in distributed mode on the cluster, so caution should be used if deviating too far from this example.
  * **utils.py** - Contians utility functions that are used in the training script.

## Example Usage

To run code locally using **train.py** (within the docker container that you've built):

```train.py --project_name=example_project --run_id=example_project_run_123 --train_list "training_lists/training.txt" --holdout_list "training_lists/holdout.txt" --img_dir "/mnt/Training_Pool/Training/example_project/data/mb_femoral_pytorch/images" --mask_dir "/mnt/Training_Pool/Training/example_project/data/mb_femoral_pytorch/masks" --checkpoint_dir "/mnt/Training_Pool/Training/example_project/runs" --epochs 100 --batch_size 8```

To run code on the cluster using **deploy.py**:

```deploy.py --project_name=example_project --mode="internal_cluster" --entrypoint ../source/train.py --train_list "training_lists/training.txt" --holdout_list "training_lists/holdout.txt" --img_dir "/mnt/Training_Pool/Training/example_project/data/mb_femoral_pytorch/images" --mask_dir "/mnt/Training_Pool/Training/example_project/data/mb_femoral_pytorch/masks" --checkpoint_dir "/mnt/Training_Pool/Training/example_project/runs" --epochs 100 --batch_size 8```

Note that when you run using deploy.py a run_id is automatically created for you.