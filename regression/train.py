import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset, custom_collate_fn

import torch
import torch.utils.data
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--model_dir",
    default="saved_models",
    help="Directory to save models",
    dest="model_dir",
)
parser.add_argument(
    "--init_model_file",
    default="",
    help="Initial model file (optional)",
    dest="init_model_file",
)
parser.add_argument(
    "--dataset_dir",
    default="",
    help="",
    dest="dataset_dir",
)
parser.add_argument(
    "--patch_size", default="32", type=int, help="Patch size", dest="patch_size"
)
parser.add_argument(
    "--num_instances",
    default="64",
    type=int,
    help="Number of instances",
    dest="num_instances",
)
parser.add_argument(
    "--num_classes", default="1", type=int, help="Number of classes", dest="num_classes"
)
parser.add_argument(
    "--batch_size", default="32", type=int, help="Batch size", dest="batch_size"
)
parser.add_argument(
    "--mil_pooling_filter",
    default="distribution",
    help="MIL pooling filter",
    dest="mil_pooling_filter",
)
parser.add_argument(
    "--num_bins", default="21", type=int, help="Number of bins", dest="num_bins"
)
parser.add_argument("--sigma", default="0.0167", type=float, help="Sigma", dest="sigma")
parser.add_argument(
    "--num_features",
    default="32",
    type=int,
    help="Number of features",
    dest="num_features",
)
parser.add_argument(
    "--learning_rate",
    default="1e-4",
    type=float,
    help="Learning rate",
    dest="learning_rate",
)
parser.add_argument(
    "--num_epochs", default=1000, type=int, help="Number of epochs", dest="num_epochs"
)
parser.add_argument(
    "--save_interval", default=50, type=int, help="Save interval", dest="save_interval"
)
parser.add_argument(
    "--metrics_file", default="loss_data", help="Metrics file", dest="metrics_file"
)

parser.add_argument(
    "--early_stopping_patience",
    default=400,
    type=int,
    help="Early stopping patience",
    dest="early_stopping_patience",
)

parser.add_argument(
    "--monitor",
    default="val_loss",
    choices=["val_loss", "val_acc"],
    help="Metric to monitor",
    dest="monitor",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
    dest="seed",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use deterministic algorithms (may be slower)",
    dest="deterministic",
)
parser.add_argument(
    "--M",
    default=42,
    type=int,
    help="Random seed for reproducibility",
    dest="M",
)
parser.add_argument(
    "--K",
    default=42,
    type=int,
    help="Random seed for reproducibility",
    dest="K",
)
parser.add_argument(
    "--T",
    default=42,
    type=int,
    help="Random seed for reproducibility",
    dest="T",
)
parser.add_argument(
    "--no_sigmoid",
    action="store_true",
    help="Random seed for reproducibility",
    dest="no_sigmoid",
)
FLAGS = parser.parse_args()
config = vars(FLAGS)


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes some CUDA ops deterministic (can impact speed/availability)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions; will error if an op has no deterministic impl
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print("Warning: torch.use_deterministic_algorithms(True) failed:", e)
    else:
        # Faster, but may be non-deterministic
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    # Ensure each dataloader worker has a distinct, reproducible seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(FLAGS.seed, deterministic=FLAGS.deterministic)
current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = f"{FLAGS.metrics_file}/step_loss_acc_metrics{current_time}.txt"

print("Model parameters:")
print(f"dataset_dir = {FLAGS.dataset_dir}")
print(f"num_classes = {FLAGS.num_classes}")
print(f"num_features = {FLAGS.num_features}")
print(f"num_instances = {FLAGS.num_instances}")
print(f"batch_size = {FLAGS.batch_size}")
print(f"mil_pooling_filter = {FLAGS.mil_pooling_filter}")
print(f"num_bins = {FLAGS.num_bins}")
print(f"sigma = {FLAGS.sigma}")
print(f"learning_rate = {FLAGS.learning_rate}")
print(f"num_epochs = {FLAGS.num_epochs}")
print(f"metrics_file = {FLAGS.metrics_file}")

train_dataset = Dataset(
    dataset_dir=FLAGS.dataset_dir,
    dataset_type="train",
    patch_size=FLAGS.patch_size,
    num_instances=FLAGS.num_instances,
)
num_images_train = train_dataset.num_images
print(f"Training Data - num_images: {num_images_train}")

val_dataset = Dataset(
    dataset_dir=FLAGS.dataset_dir,
    dataset_type="val",
    patch_size=FLAGS.patch_size,
    num_instances=FLAGS.num_instances,
)
num_images_val = val_dataset.num_images
print(f"Validation Data - num_images: {num_images_val}")

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn,
)

val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=FLAGS.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=custom_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(
    num_classes=FLAGS.num_classes,
    num_instances=FLAGS.num_instances,
    num_features=FLAGS.num_features,
    mil_pooling_filter=FLAGS.mil_pooling_filter,
    num_bins=FLAGS.num_bins,
    sigma=FLAGS.sigma,
    M=FLAGS.M,
    K=FLAGS.K,
    T=FLAGS.T,
    no_sigmoid=FLAGS.no_sigmoid,
)
model.to(device)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=FLAGS.learning_rate,
    weight_decay=0.0005,
)

best_metric = float("inf") if FLAGS.monitor == "val_loss" else -float("inf")
best_epoch = 0
no_improve_epochs = 0
patience = FLAGS.early_stopping_patience

for epoch in range(FLAGS.num_epochs):
    training_loss = 0.0
    validation_loss = 0.0
    num_predictions = 0

    model.train()
    pbar = tqdm(total=len(train_data_loader))

    for images, targets in train_data_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * targets.size(0)
        num_predictions += targets.size(0)
        pbar.update(1)

    pbar.close()
    training_loss /= num_predictions

    model.eval()
    num_predictions = 0
    pbar = tqdm(total=len(val_data_loader))

    with torch.no_grad():
        for images, targets in val_data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            validation_loss += loss.item() * targets.size(0)
            num_predictions += targets.size(0)
            pbar.update(1)

    pbar.close()
    validation_loss /= num_predictions

    print(
        f"Epoch={epoch+1} ### training_loss={training_loss:.3f} "
        f"### validation_loss={validation_loss:.3f}--- {FLAGS.mil_pooling_filter}"
    )

    current_metric = validation_loss
    improved = current_metric < best_metric - 1e-8

    if improved:
        best_metric = current_metric
        best_epoch = epoch + 1
        no_improve_epochs = 0

        best_model_filename = os.path.join(
            FLAGS.model_dir,
            f"best_model{current_time}_{FLAGS.mil_pooling_filter}_seed{FLAGS.seed}_M{FLAGS.M}_K{FLAGS.K}_T{FLAGS.T}_nosigmoid{FLAGS.no_sigmoid}.pth",
        )

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "monitor": FLAGS.monitor,
                "config": config,
            },
            best_model_filename,
        )

        print(
            f"New best model saved at epoch {best_epoch} "
            f"({FLAGS.monitor}={best_metric:.6f})"
        )
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(
            f"Early stopping triggered after {no_improve_epochs} "
            f"epochs without improvement."
        )
        break

print("Training finished!!!")

final_model_filename = (
    FLAGS.model_dir
    + "/state_dict"
    + current_time
    + f"__{epoch+1}_{str(FLAGS.seed)}.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    },
    final_model_filename,
)

print("Model weights saved in file:", final_model_filename)
