"""
Warning: the strings defined below are very sensitive with regards to 
indentation so if some indentation looks wierd, 
please don't change it as it will probably cause an error
"""

from typing import Dict, Tuple, List, AnyStr

cluster_yaml = """apiVersion: v1
kind: Config
clusters:
- name: "training-production-cluster"
  cluster:
    server: "https://iu-dl-mstr.medaphor.com:8443/k8s/clusters/c-xcvw9"
    certificate-authority-data: "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJwakNDQ\
      VUyZ0F3SUJBZ0lCQURBS0JnZ3Foa2pPUFFRREFqQTdNUnd3R2dZRFZRUUtFeE5rZVc1aGJXbGoKY\
      kdsemRHVnVaWEl0YjNKbk1Sc3dHUVlEVlFRREV4SmtlVzVoYldsamJHbHpkR1Z1WlhJdFkyRXdIa\
      GNOTWpFeApNVEF4TVRVME16VXhXaGNOTXpFeE1ETXdNVFUwTXpVeFdqQTdNUnd3R2dZRFZRUUtFe\
      E5rZVc1aGJXbGpiR2x6CmRHVnVaWEl0YjNKbk1Sc3dHUVlEVlFRREV4SmtlVzVoYldsamJHbHpkR\
      1Z1WlhJdFkyRXdXVEFUQmdjcWhrak8KUFFJQkJnZ3Foa2pPUFFNQkJ3TkNBQVFGaTY2d3crYXI4U\
      mxqRG1zZ3FJdEgzSXkwS2xsQ3k1bmVFdm5QY1k0MApBdDQwL3dEZmhnNGtsTFdSNm1zZkEyemNrV\
      DFOVWd4SmwxZnQzN3VRNmJod28wSXdRREFPQmdOVkhROEJBZjhFCkJBTUNBcVF3RHdZRFZSMFRBU\
      UgvQkFVd0F3RUIvekFkQmdOVkhRNEVGZ1FVNXFna0U4cURmODRlbHVCb1JmSXgKenZOcUIxWXdDZ\
      1lJS29aSXpqMEVBd0lEUndBd1JBSWdXdGc2TVRtd2FodG9KYWdQS1FqNm5nWkYzZkQwajdBagpuW\
      Gk0RnBNVmFEQUNJSDBTZVVMNDBMOWh2c0tnZzFPNTN4R1Y1N1BHeC95bU1rOWZtSnRjOTkxQgotL\
      S0tLUVORCBDRVJUSUZJQ0FURS0tLS0t"
- name: "training-production-cluster-iu-dlp-clus-02"
  cluster:
    server: "https://172.16.12.5:6443"
    certificate-authority-data: "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM0VENDQ\
      WNtZ0F3SUJBZ0lCQURBTkJna3Foa2lHOXcwQkFRc0ZBREFTTVJBd0RnWURWUVFERXdkcmRXSmwKT\
      FdOaE1CNFhEVEl5TURReE16QTVNRGt4TWxvWERUTXlNRFF4TURBNU1Ea3hNbG93RWpFUU1BNEdBM\
      VVFQXhNSAphM1ZpWlMxallUQ0NBU0l3RFFZSktvWklodmNOQVFFQkJRQURnZ0VQQURDQ0FRb0NnZ\
      0VCQU42NjV4ZUF3V3RBCnMzZmxHNk5OT3ZHbXNtbUQydmRFcStNcEZ3c3dDMFRlK0w3SUpjYXJ1c\
      zBDZ0hESFJVV2o5Z2Vud1dTRmNpTWkKNmprdEw0QUk3VitBdXRaRUIydGZZSy81dnYxSUtYbkI2c\
      URFVXhUR044dFhKdlFtWElJcGNRdkc4NnVmOFpPNwpyOXBJTUp4WkNXbFcwRGxtNEgzNkU0MjJmS\
      mIwOG9JaVc0bkVnczBDRG9zb0k3cjRyYXpZM3dsYzdTN0dFWEtOCmt4Z08yNGVLdEFoRThUcTZ4N\
      HpDYVFOYUx3YzZDY1Z4OVUrRGlBckJiK3RrVDh2Wll5ODVNYUIvRGxuRmVidlIKWEV3ck5QWE1RZ\
      DcvT2Ewa3Q0WjJFMytjcUNlRFdjUDdzVldZTVdacU1Bdy9TYkk2c2U2SmswY2owYmNVNFRaRApKS\
      mdDVlh4QWwyMENBd0VBQWFOQ01FQXdEZ1lEVlIwUEFRSC9CQVFEQWdLa01BOEdBMVVkRXdFQi93U\
      UZNQU1CCkFmOHdIUVlEVlIwT0JCWUVGUEJNc1RNcTJKNkpJVkthektsZ3FHQlV4OGRFTUEwR0NTc\
      UdTSWIzRFFFQkN3VUEKQTRJQkFRQzdtQ1JmY2J6MnhNNGI1QTBiM01zNWFWVkJ4U3hWSEd2SHdIb\
      nc5cXEvUVFLM1UzdVBsQXpWaVhqdApXOGtYVnA5eEZLV1g1YnYrMFVCK0xXTWtSV2R6Q0dhOFFSS\
      2JaVjZ1eTVBS3BqVm9CZVA1TDBxWFg4MWtyWnN1CkxRcTlhOWhwbWNwKzQ0eEpFVldJazFqTEsvM\
      VBtUzc5M0RaU0w4K0VMY3NQV09HZTl4MXRNT0lYd0g2VUZCZSsKWWxMWFZ0b213VkgxU2xFYSszW\
      EgvZUhYU1ZNeHNQdmkySUJmNWJlR1JsOVBqMmEwZ0U4eEVuMThYS0hpZFdwOQorQXNDR2EvM2VZS\
      jNGY05vMVNMUXdUOXhQRjJiVWZ2QXhrNzA2ZmJXL0pDTnB0R0VCWUVXZm8ySDZSRVZiN2pMCmNFV\
      HpLdk03ZXh6cklINE9mTDg3T1NRb0RmYzEKLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQo="
- name: "training-production-cluster-iu-dlp-clus-01"
  cluster:
    server: "https://172.16.12.4:6443"
    certificate-authority-data: "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM0VENDQ\
      WNtZ0F3SUJBZ0lCQURBTkJna3Foa2lHOXcwQkFRc0ZBREFTTVJBd0RnWURWUVFERXdkcmRXSmwKT\
      FdOaE1CNFhEVEl5TURReE16QTVNRGt4TWxvWERUTXlNRFF4TURBNU1Ea3hNbG93RWpFUU1BNEdBM\
      VVFQXhNSAphM1ZpWlMxallUQ0NBU0l3RFFZSktvWklodmNOQVFFQkJRQURnZ0VQQURDQ0FRb0NnZ\
      0VCQU42NjV4ZUF3V3RBCnMzZmxHNk5OT3ZHbXNtbUQydmRFcStNcEZ3c3dDMFRlK0w3SUpjYXJ1c\
      zBDZ0hESFJVV2o5Z2Vud1dTRmNpTWkKNmprdEw0QUk3VitBdXRaRUIydGZZSy81dnYxSUtYbkI2c\
      URFVXhUR044dFhKdlFtWElJcGNRdkc4NnVmOFpPNwpyOXBJTUp4WkNXbFcwRGxtNEgzNkU0MjJmS\
      mIwOG9JaVc0bkVnczBDRG9zb0k3cjRyYXpZM3dsYzdTN0dFWEtOCmt4Z08yNGVLdEFoRThUcTZ4N\
      HpDYVFOYUx3YzZDY1Z4OVUrRGlBckJiK3RrVDh2Wll5ODVNYUIvRGxuRmVidlIKWEV3ck5QWE1RZ\
      DcvT2Ewa3Q0WjJFMytjcUNlRFdjUDdzVldZTVdacU1Bdy9TYkk2c2U2SmswY2owYmNVNFRaRApKS\
      mdDVlh4QWwyMENBd0VBQWFOQ01FQXdEZ1lEVlIwUEFRSC9CQVFEQWdLa01BOEdBMVVkRXdFQi93U\
      UZNQU1CCkFmOHdIUVlEVlIwT0JCWUVGUEJNc1RNcTJKNkpJVkthektsZ3FHQlV4OGRFTUEwR0NTc\
      UdTSWIzRFFFQkN3VUEKQTRJQkFRQzdtQ1JmY2J6MnhNNGI1QTBiM01zNWFWVkJ4U3hWSEd2SHdIb\
      nc5cXEvUVFLM1UzdVBsQXpWaVhqdApXOGtYVnA5eEZLV1g1YnYrMFVCK0xXTWtSV2R6Q0dhOFFSS\
      2JaVjZ1eTVBS3BqVm9CZVA1TDBxWFg4MWtyWnN1CkxRcTlhOWhwbWNwKzQ0eEpFVldJazFqTEsvM\
      VBtUzc5M0RaU0w4K0VMY3NQV09HZTl4MXRNT0lYd0g2VUZCZSsKWWxMWFZ0b213VkgxU2xFYSszW\
      EgvZUhYU1ZNeHNQdmkySUJmNWJlR1JsOVBqMmEwZ0U4eEVuMThYS0hpZFdwOQorQXNDR2EvM2VZS\
      jNGY05vMVNMUXdUOXhQRjJiVWZ2QXhrNzA2ZmJXL0pDTnB0R0VCWUVXZm8ySDZSRVZiN2pMCmNFV\
      HpLdk03ZXh6cklINE9mTDg3T1NRb0RmYzEKLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQo="

users:
- name: "training-production-cluster"
  user:
    token: "kubeconfig-user-wwmpdqln5k:2cwmcvddrxssqjwftjv2gv5st7h4v84p4cwrl2mqwmd9j9f8bm97ct"


contexts:
- name: "training-production-cluster"
  context:
    user: "training-production-cluster"
    cluster: "training-production-cluster"

current-context: "training-production-cluster"
"""


def fill_job_yaml(run_id, max_replicas, entrypoint, yaml_args_str, wandb_api_key, project_name, priority):
    job_yaml = f"""apiVersion: elastic.pytorch.org/v1alpha1
kind: ElasticJob
metadata:
  name: {run_id}
  namespace: elastic-job
  labels:
    priority: "{priority}"
spec:
  RunPolicy:
    cleanPodPolicy: None
  maxReplicas: {max_replicas}
  minReplicas: 1
  rdzvEndpoint: "etcd-service:2379"
  replicaSpecs:
    Worker:
      #the number of pods to start with
      replicas: 1
      restartPolicy: ExitCode
      template:
        metadata:
          creationTimestamp: null
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
                    - iu-dlp-node-01
          containers:
          - args:
            # this should be equal to the number of GPUs per node
            - --nproc_per_node=1
            - --nnodes=1:{max_replicas}
            # if you have more than one job deployed at the same time this value must be unique for each job
            - --rdzv_id={run_id}
            - {entrypoint}
            - {yaml_args_str}
            command: ["python", "-m", "torch.distributed.run"]
            image: git.medaphor.com:5050/iudl/dl-training-docker-images/{project_name}
            env:
              - name: WANDB_API_KEY
                value: {wandb_api_key}
            imagePullPolicy: Always
            name: elasticjob-worker
            resources:
              limits:
                nvidia.com/gpu: "1"
                memory: 12Gi
                cpu: 4
            volumeMounts:
            - mountPath: /mnt/Training_Pool/Training
              name: training-storage-pv
            - mountPath: /dev/shm
              name: dshm
          imagePullSecrets:
            - name: regcred
          volumes:
          - name: training-storage-pv
            persistentVolumeClaim:
              claimName: training-storage-pvc
          - emptyDir:
              medium: Memory
            name: dshm"""

    return job_yaml


def fill_yaml_args(config: Dict) -> Tuple[List, AnyStr]:
    """
    creates a list of params and string to populate yaml file

    Paramaters
    ----------
    config: dictionary containing command line
    parameters as keys and their values as values

    Returns
    -------
    args_list: list of format [--command1:value1, command2:value2 ... commandn:valuen]
    yaml_args_str: string that can then be used to populate torchelastic job launch yaml file
    """

    args_list = [f"--{k}={v}" for k, v in config.items()]
    yaml_args_str = "\n            - ".join([f"--{k}={v}" for k, v in config.items()])

    return args_list, yaml_args_str

