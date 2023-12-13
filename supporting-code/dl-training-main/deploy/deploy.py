import subprocess
import time
import os
import logging
import yaml
from kubernetes import client
from kubernetes.client import Configuration
from kubernetes.config import kube_config
from templates import cluster_yaml, fill_job_yaml, fill_yaml_args
import argparse

logging.basicConfig(level=logging.INFO,  handlers=[logging.FileHandler("deploy.log"),logging.StreamHandler()])


class K8s(object):
    def __init__(self, configuration_yaml):
        self.configuration_yaml = configuration_yaml
        self._configuration_yaml = None

    @property
    def config(self):
        self._configuration_yaml = yaml.safe_load(self.configuration_yaml)
        return self._configuration_yaml

    @property
    def client(self):
        k8_loader = kube_config.KubeConfigLoader(self.config)
        call_config = type.__call__(Configuration)
        k8_loader.load_and_set(call_config)
        Configuration.set_default(call_config)
        return client


def main(deploy_args, train_args):

    env = os.environ.copy()

    if not deploy_args.project_name.isalnum():
        logging.error("project_name contains non alphanumeric characters, please try again and enter valid jobid")
        return
    if deploy_args.project_name == "":
        logging.error("You haven't enterd a project name, please try again")
        return
    if deploy_args.project_name.lower() != deploy_args.project_name:
        logging.error("Some of the characters in your project ID are uppercase which arent allowed, please try again")
        return
    run_id = deploy_args.project_name + f"{time.strftime('%Y%m%d%H%M%S')}"
    train_args_dict = vars(train_args)

    logging.info(f"mode: {deploy_args.mode}")
    logging.info(f"build_only: {deploy_args.build_only}")
    logging.info(f"project_name: {deploy_args.project_name}")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Entrypoint: {deploy_args.entrypoint}")
    logging.info(f"Script args:")
    for k, v in train_args_dict.items():
        logging.info(f"  {k}: {v}")
    print(f"deploy_args.project_name: {deploy_args.project_name}")
    image_pth = f"git.medaphor.com:5050/iudl/dl-training-docker-images/{deploy_args.project_name}"
    logging.info("Building and pushing docker image")
    try:
        env["PROJECT_NAME"] = deploy_args.project_name
        subprocess.run(["docker-compose", "build"], cwd="../deploy", env=env)
        subprocess.run(["docker", "tag", deploy_args.project_name, image_pth])
        subprocess.run(["docker", "push", image_pth])
        # run docker container from built image
        logging.info("docker image built and pushed")
    except Exception as e:
        logging.error("pushing docker image unsucccessful : " + str(e))
        return

    train_args_dict["project_name"] = deploy_args.project_name

    if deploy_args.resume_run_id:
        train_args_dict["run_id"] = deploy_args.resume_run_id
    else:
        train_args_dict["run_id"] = run_id

    train_args_dict["wandb_api_key"] = deploy_args.wandb_api_key

    args_list, yaml_args_str = fill_yaml_args(train_args_dict)

    if not deploy_args.build_only:

        # train on local machine
        if deploy_args.mode == "local_machine":
            logging.info("deploying on local machine")

            try:

                subprocess.run(["docker-compose", "run", "pytorch", f"{deploy_args.entrypoint}"] + args_list, env=env)
            except Exception as e:
                logging.error(f"failed to run docker container : " + str(e))
                return

        # train on internally hosted cluster
        elif deploy_args.mode == "internal_cluster":
            logging.info("deploying on internal cluster")
            job_yaml = fill_job_yaml(run_id, 6, deploy_args.entrypoint, yaml_args_str,
                                     os.environ.get("WANDB_API_KEY"), deploy_args.project_name, 1)
            k8s_client = K8s(configuration_yaml=cluster_yaml).client
            _ = k8s_client.CustomObjectsApi().create_namespaced_custom_object(group="elastic.pytorch.org",
                                                                              version="v1alpha1",
                                                                              namespace="elastic-job",
                                                                              plural="elasticjobs",
                                                                              body=yaml.safe_load(job_yaml))

        # train on AWS cluster
        elif deploy_args.mode == "cloud_cluster":
            raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    deploy_parser = argparse.ArgumentParser()
    deploy_parser.add_argument('--entrypoint', type=str, required=True, help="Project Name")
    deploy_parser.add_argument('--mode', required=True, choices=["local_machine", "internal_cluster"]),
    deploy_parser.add_argument('--project_name', type=str, required=True, help="Project Name")
    deploy_parser.add_argument('--wandb_api_key', type=str, required=True, default=None)
    deploy_parser.add_argument('--build_only', action='store_true')
    deploy_parser.add_argument('--resume_run_id', type=str, default=None, required=False, help="resume Run ID")

    deploy_args, unknown = deploy_parser.parse_known_args()

    train_parser = argparse.ArgumentParser()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            train_parser.add_argument(arg.split('=')[0])

    train_args, unknown = train_parser.parse_known_args()


    main(deploy_args, train_args)
