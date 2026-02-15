import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time
import random
from model import Model
from dataset import Dataset, custom_collate_fn

import torch
import torch.utils.data

from tqdm import tqdm


from torch.utils.data import DataLoader, Subset

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
    "--num_classes", default="2", type=int, help="Number of classes", dest="num_classes"
)
parser.add_argument(
    "--batch_size", default="32", type=int, help="Batch size", dest="batch_size"
)
parser.add_argument(
    "--mil_pooling_filter",
    default="distribution",
    help="MIL pooling filter: distribution, mean, attention, max",
    dest="mil_pooling_filter",
)
parser.add_argument(
    "--num_bins",
    default="21",
    type=int,
    help="Number of bins in distribution pooling filters",
    dest="num_bins",
)
parser.add_argument(
    "--sigma",
    default="0.0167",
    type=float,
    help="sigma in Gaussian kernel in distribution pooling filters",
    dest="sigma",
)
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
    "--num_epochs", default=2000, type=int, help="Number of epochs", dest="num_epochs"
)
parser.add_argument(
    "--metrics_file",
    default="loss_data",
    help="Text file to write step, loss, accuracy metrics",
    dest="metrics_file",
)

# Early stopping args
parser.add_argument(
    "--early_stopping_patience",
    default=400,
    type=int,
    help="Early stopping patience (epochs without improvement)",
    dest="early_stopping_patience",
)
parser.add_argument(
    "--monitor",
    default="val_loss",
    choices=["val_loss", "val_acc"],
    help="Which metric to monitor for early stopping and best model saving (val_loss or val_acc)",
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
metrics_file = "{}/step_loss_acc_metrics{}.txt".format(FLAGS.metrics_file, current_time)

# Ensure model dir exists
os.makedirs(FLAGS.model_dir, exist_ok=True)

print("Model parameters:")
print("dataset_dir = {}".format(FLAGS.dataset_dir))
print("num_classes = {}".format(FLAGS.num_classes))
print("num_features = {}".format(FLAGS.num_features))
print("num_instances = {}".format(FLAGS.num_instances))
print("batch_size = {}".format(FLAGS.batch_size))
print("mil_pooling_filter = {}".format(FLAGS.mil_pooling_filter))
print("num_bins = {}".format(FLAGS.num_bins))
print("sigma = {}".format(FLAGS.sigma))
print("learning_rate = {}".format(FLAGS.learning_rate))
print("num_epochs = {}".format(FLAGS.num_epochs))
print("metrics_file = {}".format(FLAGS.metrics_file))
print("early_stopping_patience = {}".format(FLAGS.early_stopping_patience))
print("monitor = {}".format(FLAGS.monitor))


train_dataset = Dataset(
    dataset_dir=FLAGS.dataset_dir,
    dataset_type="train",
    patch_size=FLAGS.patch_size,
    num_instances=FLAGS.num_instances,
)
num_images_train = train_dataset.num_images


N = 32
# rng = np.random.RandomState(0)
# small_idx = rng.choice(len(train_dataset), size=N, replace=False)

# small_train_ds = Subset(train_dataset, small_idx)

print("Training Data - num_images: {}".format(num_images_train))

val_dataset = Dataset(
    dataset_dir=FLAGS.dataset_dir,
    dataset_type="val",
    patch_size=FLAGS.patch_size,
    num_instances=FLAGS.num_instances,
)
num_images_val = val_dataset.num_images
print("Validation Data - num_images: {}".format(num_images_val))

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

# define loss criterion
criterion = torch.nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=0.0005)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=device)
        # try to restore both model and optimizer if available
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        if "optimizer_state_dict" in state_dict:
            try:
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except Exception as e:
                print("Warning: couldn't load optimizer state: ", e)
        print("weights loaded successfully!!!\n{}".format(FLAGS.init_model_file))

with open(metrics_file, "w") as f_metric_file:
    f_metric_file.write("# Model parameters:\n")
    f_metric_file.write("# dataset_dir = {}\n".format(FLAGS.dataset_dir))
    f_metric_file.write("# patch_size = {}\n".format(FLAGS.patch_size))
    f_metric_file.write("# num_instances = {}\n".format(FLAGS.num_instances))
    f_metric_file.write("# num_classes = {}\n".format(FLAGS.num_classes))
    f_metric_file.write("# batch_size = {}\n".format(FLAGS.batch_size))
    f_metric_file.write("# learning_rate = {}\n".format(FLAGS.learning_rate))
    f_metric_file.write("# Training Data - num_images: {}\n".format(num_images_train))
    f_metric_file.write("# Validation Data - num_images: {}\n".format(num_images_val))
    f_metric_file.write("# mil_pooling_filter = {}\n".format(FLAGS.mil_pooling_filter))
    f_metric_file.write("# num_bins = {}\n".format(FLAGS.num_bins))
    f_metric_file.write("# sigma = {}\n".format(FLAGS.sigma))
    f_metric_file.write("# num_features = {}\n".format(FLAGS.num_features))
    f_metric_file.write("# metrics_file = {}\n".format(FLAGS.metrics_file))
    f_metric_file.write("# model_dir: {}\n".format(FLAGS.model_dir))
    f_metric_file.write("# init_model_file: {}\n".format(FLAGS.init_model_file))
    f_metric_file.write("# num_epochs: {}\n".format(FLAGS.num_epochs))
    # f_metric_file.write('# save_interval = {}\n'.format(FLAGS.save_interval))
    f_metric_file.write(
        "# early_stopping_patience = {}\n".format(FLAGS.early_stopping_patience)
    )
    f_metric_file.write("# monitor = {}\n".format(FLAGS.monitor))
    f_metric_file.write(
        "# epoch\ttraining_acc\ttraining_loss\tvalidation_acc\tvalidation_loss\n"
    )

# Early stopping / best model tracking initialization
if FLAGS.monitor == "val_loss":
    best_metric = float("inf")  # lower is better
else:  # 'val_acc'
    best_metric = -float("inf")  # higher is better

best_epoch = -1
no_improve_epochs = 0
patience = FLAGS.early_stopping_patience

for epoch in range(FLAGS.num_epochs):
    training_loss = 0.0
    validation_loss = 0.0
    gmm_stats_train = None

    # train for one epoch
    num_corrects = 0
    num_predictions = 0

    pbar = tqdm(
        total=len(train_data_loader),
        desc=f"Train Epoch {epoch+1}/{FLAGS.num_epochs}",
        unit="batch",
    )

    model.train()
    for images, targets in train_data_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if FLAGS.mil_pooling_filter == "gmm" and gmm_stats_train is None:
            y_logits, gmm_stats_train = model(images, return_gmm_bag_embedding=True)
        else:
            y_logits = model(images)
        loss = criterion(y_logits, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * targets.size(0)

        num_predictions += targets.size(0)

        predicted_labels = torch.argmax(y_logits, dim=1)
        correct_predictions = torch.sum(predicted_labels == targets)
        num_corrects += correct_predictions.item()

        pbar.update(1)

    training_loss /= num_predictions if num_predictions > 0 else 1.0
    training_acc = num_corrects / num_predictions if num_predictions > 0 else 0.0

    pbar.close()

    # evaluate on the validation dataset
    num_corrects = 0
    num_predictions = 0
    gmm_stats_val = None

    pbar = tqdm(
        total=len(val_data_loader),
        desc=f"Val Epoch {epoch+1}/{FLAGS.num_epochs}",
        unit="batch",
    )

    model.eval()
    val_target_counts = torch.zeros(FLAGS.num_classes, device=device)
    val_pred_counts = torch.zeros(FLAGS.num_classes, device=device)
    with torch.no_grad():
        for images, targets in val_data_loader:
            images = images.to(device)
            targets = targets.to(device)

            if FLAGS.mil_pooling_filter == "gmm" and gmm_stats_val is None:
                y_logits, gmm_stats_val = model(images, return_gmm_bag_embedding=True)
            else:
                y_logits = model(images)
            loss = criterion(y_logits, targets)
            assert targets.dtype == torch.long, targets.dtype
            assert targets.dim() == 1, targets.shape
            validation_loss += loss.item() * targets.size(0)
            num_predictions += targets.size(0)
            predicted_labels = torch.argmax(y_logits, dim=1)
            correct_predictions = torch.sum(predicted_labels == targets)
            num_corrects += correct_predictions.item()
            for c in range(FLAGS.num_classes):
                val_target_counts[c] += (targets == c).sum()
                val_pred_counts[c] += (predicted_labels == c).sum()
            pbar.update(1)

    validation_loss /= num_predictions if num_predictions > 0 else 1.0
    validation_acc = num_corrects / num_predictions if num_predictions > 0 else 0.0

    pbar.close()
    # print("VAL targets:", val_target_counts.long().cpu().numpy())
    # print("VAL preds  :", val_pred_counts.long().cpu().numpy())
    # if FLAGS.mil_pooling_filter == "gmm":
    #     def fmt_stats(s):
    #         if s is None: return "None"
    #         pis = s["pis_mean"].cpu().numpy()
    #         return (
    #             f"logp_std_mean={s['logp_std_mean'].item():.4f} | "
    #             f"sigmas_mean={s['sigmas_mean'].item():.4f} "
    #             f"(min={s['sigmas_min'].item():.4f}, max={s['sigmas_max'].item():.4f}) | "
    #             f"pis_mean={np.round(pis, 3)}"
    #         )

    #     print("GMM train stats:", fmt_stats(gmm_stats_train))
    #     print("GMM   val stats:", fmt_stats(gmm_stats_val))

    print(
        "Epoch=%d ### training_acc=%5.3f, training_loss=%5.3f ### validation_acc=%5.3f, validation_loss=%5.3f"
        % (epoch + 1, training_acc, training_loss, validation_acc, validation_loss)
    )

    with open(metrics_file, "a") as f_metric_file:
        f_metric_file.write(
            "%d\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n"
            % (epoch + 1, training_acc, training_loss, validation_acc, validation_loss)
        )

    # Determine current metric value depending on monitor
    if FLAGS.monitor == "val_loss":
        current_metric = validation_loss
        improved = (
            current_metric < best_metric - 1e-8
        )  # small epsilon to avoid float noise
    else:  # val_acc
        current_metric = validation_acc
        improved = current_metric > best_metric + 1e-8

    # Save best model if improved
    if improved:
        best_metric = current_metric
        best_epoch = epoch + 1
        no_improve_epochs = 0

        best_model_filename = os.path.join(
            FLAGS.model_dir,
            f"best_model{current_time}_{FLAGS.mil_pooling_filter}_seed{FLAGS.seed}_M{FLAGS.M}_K{FLAGS.K}_T{FLAGS.T}_nosigmoid{FLAGS.no_sigmoid}.pth",
        )
        state_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "monitor": FLAGS.monitor,
            "config": config,
        }
        torch.save(state_dict, best_model_filename)
        print(
            f"New best model (epoch {epoch+1}) saved to: {best_model_filename} (best {FLAGS.monitor} = {best_metric:.6f})"
        )
    else:
        no_improve_epochs += 1
        # print(
        #     f"No improvement in {FLAGS.monitor} for {no_improve_epochs} epoch(s) (patience = {patience})"
        # )

    # Early stopping check
    if no_improve_epochs >= patience:
        print(
            f"Early stopping triggered. No improvement in {FLAGS.monitor} for {no_improve_epochs} epochs (patience={patience})."
        )
        break

print("Training finished!!!")

# final save (if last epoch did not produce best, we still save the last state)
final_model_filename = os.path.join(
    FLAGS.model_dir,
    "final_state_dict"
    + current_time
    + "_"
    + FLAGS.mil_pooling_filter
    + "_"
    + str(epoch + 1)
    + "seed"
    + str(FLAGS.seed)
    + ".pth",
)
state_dict = {
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
}
torch.save(state_dict, final_model_filename)
print("Final model weights saved in file: ", final_model_filename)

# Summarize best model location if any
if best_epoch != -1:
    print(
        f"Best model was at epoch {best_epoch} with {FLAGS.monitor} = {best_metric:.6f}. Saved in {os.path.join(FLAGS.model_dir, f'best_model{current_time}.pth')}"
    )
else:
    print("No best model was saved during training (no improvements were detected).")
