import argparse
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import DataLoader, Subset

import wandb
from dataloader import MyDataset

from UNet_Version_master.models.UNet_3Plus_Halved import UNet_3Plus_DeepSup as UNet
from UNet_Version_master.loss.iouLoss import IOU
from UNet_Version_master.loss.msssimLoss import MSSSIM

from utils import MetricLogger, CollectSamplesToRank0, class_ious, my_random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from UNet_Version_master.loss.jaccardLoss import JaccardLoss

# replace below params with actuals params so tied in with rest of code

MASK_LABELS = ["Background", "Ribs", "Pleura", "A-Lines", "Confluent B-Lines", "Consolidations", "Effusions"]
NUM_MASK_CLASSES = len(MASK_LABELS)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

@record
def train(run_id: str,
          img_dir: str,
          mask_dir: str,
          train_list: str,
          holdout_list: str,
          batch_size: int,
          checkpoint_dir: str,
          initial_model_weights_path: str,
          epochs: int,
          learning_rate: float) -> None:
    """
    Instantiates dataloaders, transforms and model then trains model, 
    contains for loop for each epoch

    Paramaters
    ----------
    img_dir: path to folder containing images
    mask_dir: path to folder containing masks
    train_list: path to text file containing image file paths for training
    holdout_list: path to text file containing image file paths for validation
    batch_size: size of batch
    checkpoint_dir: filepath to folder where checkpoint directory will be written to
    initial_model_weights_path: path to a model checkpoint that the model will be initialised from

    """

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=25, p=1.0),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            # A.GaussNoise(var_limit=20, p=1),
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    holdout_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    fullDataset = MyDataset(
            img_dir=img_dir,
            mask_dir=mask_dir,
            transform=train_transform,
            holdout_list=holdout_list
            )
    print("full dataset length:")
    print(len(fullDataset))

    # dataset to use to specify entire patients to add to a holdout set
    holdout_dataset = MyDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=holdout_transform,
        holdout_set=True,
        holdout_list=holdout_list
    )

    print("holdout dataset length:")
    print(len(holdout_dataset))

    random_seed = 42
    train_dataset, val_dataset = my_random_split(
        fullDataset,
        [0.9, 0.1],
        # added default generator for repeatability
        torch.Generator().manual_seed(random_seed)
    )

    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, sampler=train_sampler)
    val_sampler = ElasticDistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, sampler=val_sampler)

    ckpt_path = os.path.join(checkpoint_dir, run_id, "last.pt")

    samples_loader = DataLoader(Subset(val_dataset, random.sample(range(len(val_dataset)), batch_size)),
                                shuffle=False, batch_size=batch_size, num_workers=10)

    holdout_sampler = ElasticDistributedSampler(holdout_dataset)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, num_workers=10, sampler=holdout_sampler)

    model = UNet(in_channels=1, n_classes=NUM_MASK_CLASSES)

    # DistributedDataParallel will use all available devices.
    model.to(device)
    if torch.cuda.is_available():
        model = DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
    else:
        model = DataParallel(model, device_ids=[device_id], output_device=[device_id])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       learning_rate,
                                                       epochs=epochs,
                                                       steps_per_epoch=len(train_loader),
                                                       pct_start=0.05,
                                                       div_factor=100.0,
                                                       final_div_factor=10000.0
                                                       )

    if os.path.exists(ckpt_path):
        print("Resuming training from last checkpoint")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        epochs_completed = checkpoint['epoch']
    elif initial_model_weights_path is not None and os.path.exists(initial_model_weights_path):
        print(f"Initialising model training from {initial_model_weights_path}")
        checkpoint = torch.load(initial_model_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epochs_completed = 0
    else:
        epochs_completed = 0

    cudnn.benchmark = True
    if dist.get_rank() == 0:
        wandb.watch(model)

    for epoch in range(epochs_completed + 1, epochs):
        train_one_epoch(epoch, train_loader, model, optimizer, lr_scheduler, device_id)
        print("---------------------------------------------")
        val_one_epoch(epoch, val_loader, model, device_id)
        print("---------------------------------------------")

        # added code to re-randomise the train-val datasets, remove if no holdout
        if len(holdout_dataset) > 0:

            train_dataset, val_dataset = my_random_split(
                fullDataset,
                [0.9, 0.1],
                # added default generator for repeatability
                torch.Generator().manual_seed(random_seed + epochs_completed + 1)
            )

            train_sampler = ElasticDistributedSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, sampler=train_sampler)
            val_sampler = ElasticDistributedSampler(val_dataset)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, sampler=val_sampler)

            # code to handle logging a holdout set to wandb
            holdout_one_epoch(epoch, holdout_loader, model, device_id)
            print("---------------------------------------------")

        log_samples_one_epoch(epoch, samples_loader, model)
        if dist.get_rank() == 0:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, ckpt_path)
    print("Done!")


def shared_step(batch: torch.tensor, model: nn.Module, device_id: int) -> Tuple:
    """
    Calculates losses and class intersection-over-unions, 
    so called shared because this needs to 
    happen in both train and validation loops

    Parameters
    ----------
    batch: Batch of data to be used 
    model: model for generating predictions and losses

    Returns
    -------
    tuple of losses
    """
    foreground_classes = list(range(1, NUM_MASK_CLASSES))

    msssimloss = MSSSIM().to(device)
    jaccardloss = JaccardLoss('multilabel').to(device)

    # alpha values should align with the classes:
    # ["Background", "Ribs", "Pleura", "A-Lines", "Confluent B-Lines", "Consolidations", "Effusions"]
    alpha_values = [0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]

    focalloss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=torch.tensor(alpha_values),
        gamma=2,
        reduction='mean',
        force_reload=False
    )
    focalloss = focalloss.to(device)

    x, y = batch
    x = x.to(device)
    y = y.to(device)
    pred = model(x)

    y = y.type(torch.long)
    y_one_hot = nn.functional.one_hot(y, NUM_MASK_CLASSES)
    y_one_hot = y_one_hot.permute(0, 3, 1, 2)
    y_one_hot = y_one_hot.type(torch.long)

    # condition to handle additional option of deeply supervised learning
    if torch.is_tensor(pred):
        print("No Deep Supervision")
        loss = jaccardloss(pred, y_one_hot) + focalloss(pred, y) + (
                1 - msssimloss(pred.type(torch.float32), y_one_hot.type(torch.float32)))

        iou_vals = class_ious(y, pred, class_idx_list=foreground_classes)

    else:
        print("Deep Supervision")
        avg_pred = torch.sum(torch.stack(pred), dim=0)/len(pred)

        loss = sum(
            [
                jaccardloss(p, y_one_hot)
                + focalloss(p, y)
                + (1 - msssimloss(p.type(torch.float32), y_one_hot.type(torch.float32)))
                for p in pred
            ]
        )

        iou_vals = class_ious(y, avg_pred, class_idx_list=foreground_classes)

    return loss, iou_vals


def train_one_epoch(epoch, dataloader, model, optimizer, lr_scheduler, device_id) -> None:
    """
    Does one epoch of training

    Parameters
    ----------
    epoch: Epoch number
    dataloader: pytorch dataloader, generates data used for training
    model: pytorch nn.Module model for training
    optimizer: torch.optim
    lr_scheduler: Learning rate scheduler
    device_id: CUDA device ID

    """
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Train Epoch: [{}]'.format(epoch)
    for batch, (Xy) in enumerate(metric_logger.log_every(dataloader, 100, header)):

        loss, iou_vals = shared_step(Xy, model, device_id)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for i, class_label in enumerate(MASK_LABELS[1:]):
            metric_logger.update(**{f"train_iou - {class_label}": iou_vals[:, i + 1].mean().item()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if dist.get_rank() == 0:
        epoch_averages = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        epoch_averages.update({"epoch": epoch})
        wandb.log(epoch_averages)
        wandb.log({'train_loss': epoch_averages["loss"], "epoch": epoch})
        wandb.log({'num_gpus': int(os.environ.get("WORLD_SIZE", -1)), "epoch": epoch})
        wandb.log({'lr': optimizer.param_groups[0]["lr"], "epoch": epoch})


def val_one_epoch(epoch, dataloader, model, device_id) -> None:
    """
    Calculates and logs metrics against validation data

    Parameters
    ----------
    dataloader: pytorch dataloader, generates data used for validating against
    model: pytorch nn.Module model for validating
    """

    model.eval()
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Val Epoch: [{}]'.format(26)
    with torch.no_grad():
        for batch, (Xy) in enumerate(metric_logger.log_every(dataloader, 100, header)):

            loss, iou_vals = shared_step(Xy, model, device_id)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metric_logger.update(val_loss=loss.item())
            for i, class_label in enumerate(MASK_LABELS[1:]):
                metric_logger.update(**{f"val_iou - {class_label}": iou_vals[:, i + 1].mean().item()})
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        if dist.get_rank() == 0:
            epoch_averages = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            epoch_averages.update({"epoch": epoch})
            wandb.log(epoch_averages)
            wandb.log({'val_loss': epoch_averages["val_loss"], "epoch": epoch})


def holdout_one_epoch(epoch, dataloader, model, device_id) -> None:
    """
    Calculates and logs metrics against holdout data

    Added to function as a holdout dataset test

    Parameters
    ----------
    dataloader: pytorch dataloader, generates data used for validating against
    model: pytorch nn.Module model for validating
    """

    model.eval()
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Holdout Epoch: [{}]'.format(26)
    with torch.no_grad():
        for batch, (Xy) in enumerate(metric_logger.log_every(dataloader, 100, header)):

            loss, iou_vals = shared_step(Xy, model, device_id)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metric_logger.update(holdout_loss=loss.item())
            for i, class_label in enumerate(MASK_LABELS[1:]):
                metric_logger.update(**{f"holdout_iou - {class_label}": iou_vals[:, i + 1].mean().item()})
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        if dist.get_rank() == 0:
            epoch_averages = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            epoch_averages.update({"epoch": epoch})
            wandb.log(epoch_averages)
            wandb.log({'holdout_loss': epoch_averages["holdout_loss"], "epoch": epoch})


def log_samples_one_epoch(epoch, dataloader, model) -> None:
    """
    Generate actual and predicted segmentation masks and log them to W&B

    Parameters
    ----------
    epoch: epoch number
    dataloader: pytorch dataloader
    model: pytorch nn.Module model to generate predictions
    """

    class_dict = {i: l for i, l in enumerate(MASK_LABELS)}

    pred_examples = []
    gather_x_samples = CollectSamplesToRank0()
    gather_y_samples = CollectSamplesToRank0()
    gather_p_samples = CollectSamplesToRank0()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # added new code for deep supervision
            output = model(x)
            if torch.is_tensor(output):
                preds = torch.argmax(output, dim=1)
            else:
                avg_pred = torch.sum(torch.stack(output), dim=0) / len(output)
                preds = torch.argmax(avg_pred, dim=1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gather_x_samples.update(x)
            gather_y_samples.update(y)
            gather_p_samples.update(preds)

        all_x = gather_x_samples.get_values()
        all_y = gather_y_samples.get_values()
        all_p = gather_p_samples.get_values()

        if dist.get_rank() == 0:

            for i in range(all_x.shape[0]):

                pred_mask = all_p[i].numpy()
                act_mask = all_y[i].numpy()

                saved_mask = all_x[i].permute(1, 2, 0)

                caption = f"Example {i + 1}\n"

                pred_examples.append(wandb.Image(saved_mask.numpy(), masks={
                    "actuals": {
                        "mask_data": act_mask,
                        "class_labels": class_dict
                    },
                    "predictions": {
                        "mask_data": pred_mask,
                        "class_labels": class_dict
                    }
                }, caption=caption))

            wandb.log({f"Segmentation Predictions": pred_examples, "epoch": epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True, help="Project name")
    parser.add_argument('--wandb_tags', type=str, required=True, help='Specify wandb tags - comma seperated')
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--run_id', type=str, required=True, help="Run ID")
    parser.add_argument('--img_dir', type=str, required=True, help="Location of image directory")
    parser.add_argument('--mask_dir', type=str, required=True, help="Location of mask directory")
    parser.add_argument('--train_list', type=str, required=True, help="Location of training list file")
    parser.add_argument('--holdout_list', type=str, required=True, help="Location of holdout list file")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size (per GPU)")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument('--initial_model_weights_path', type=str, required=False,
                        help="A file containing weights to initialise from")
    parser.add_argument('--epochs', type=int, required=True, help="number of epochs to train for")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--local_rank', type=int, default=0,
                        help="NO NEED TO SET - HANDLED BY torch.distributed.launch")

    args = parser.parse_args()

    if args.wandb_api_key is not None:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    rank = int(os.environ.get("RANK", -1))
    device_id = int(os.environ.get("LOCAL_RANK", -1))

    device = f"cuda:{str(device_id)}" if torch.cuda.is_available() else f"cpu:{str(device_id)}"

    Path("/wandb").mkdir(exist_ok=True)
    os.environ["WANDB_DIR"] = "/wandb"

    if rank >= 0:
        if rank == 0:
            wandb.init(config=vars(args),
                       tags=args.wandb_tags.split(","), project=args.project_name, id=args.run_id,
                       entity='tapp', resume="allow")

        print(f"rank of this worker at start of __main__ is {rank}")

        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=30))
        else:
            torch.device(device)
            dist.init_process_group(backend="gloo", init_method="env://", timeout=timedelta(seconds=30))


    else:
        rank = 0
        device_id = 0
        wandb.init(config=vars(args),
                   tags=args.wandb_tags.split(","), project=args.project_name, id=args.run_id,
                   entity='intelligent-ultrasound', resume="allow")
        dist.init_process_group(backend="gloo",
                                init_method='tcp://0.0.0.0:23456',
                                rank=0,
                                world_size=1,
                                timeout=timedelta(seconds=30))

    del args.local_rank
    del args.project_name
    del args.wandb_tags
    del args.wandb_api_key
    train(**vars(args))
