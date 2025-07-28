# # built-in modules
import os
import argparse
from pprint import pformat
from collections import OrderedDict
import random
# # Torch modules
import torch
from torch.utils.data import random_split
from torchvision import transforms, datasets
# # internal imports
from prelude import save_dicts, startup_folders, get_device, save_results_to_csv
from src.composer import Broken_CIFAR, Cued_Scattered_CIFAR, Search_CIFAR, Scattered_CIFAR, Cued_CIFAR
from src.model import AttentionModel
from src.utils import plot_all, plot_loss_all
from src.utils import build_loaders, get_n_parameters
from src.conductor import AttentionTrain

# # reproducibility
torch.manual_seed(1821)  # Hermann von Helmholtz (1821)
random.seed(1821)

parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-l2', type=float, default=1e-4)
parser.add_argument('-nc', type=int, default=1)
parser.add_argument('-exase', type=str, default="default")
parser.add_argument('-verbose', type=int, default=1)
argus = parser.parse_args()
data_path = r"./data"
train_params = {
    "n_epochs": argus.n_epochs,
    "batch_size": argus.batch_size,
    "lr": argus.lr,
    "l2": argus.l2,
    "exase": argus.exase,
    "dir": r"./results",
    "milestones": [96, 112],
    "gamma": 0.2,
    "max_grad_norm": 10.0,
    "mask_mp": 0.0,
    "scheduler": "OneCycleLR",  # OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR
    "optimizer": "Adam",  # "SGD", "Adam",
    "lr_min": 1e-4,
    "slow_soft": False,
    "sequential": False,
}
channels = [
    (3, 16, 16, 32, 32, 64, 64),
    (3, 32, 32, 64, 64, 128, 128),
    (3, 32, 64, 64, 128, 128, 256),
    (3, 64, 64, 128, 128, 256, 256),
    ][argus.nc]

model_params = {
    "in_dims": (3, 192, 192),  # input dimensions (channels, height, width)
    "n_classes": 10,  # number of classes
    "out_dim": 10,  # output dimensions (could be larger than n_classes)
    "normalize": True,  # normalize input images
    "softness": 0.5,  # softness of the attention (scale)
    "channels": channels,  # channels in the encoder
    "residuals": False,  # use residuals in the encoder
    "kernels": 3,  # kernel size
    "strides": 1,  # stride
    "paddings": 1,  # padding
    "conv_bias": True,  # bias in the convolutions
    "conv_norms": (None, *("layer" for _ in range(5))),  # normalization in the encoder
    "conv_dropouts": 0.0,  # dropout in the encoder
    "conv_funs": torch.nn.GELU(),  # activation function in the encoder
    "deconv_funs": torch.nn.Tanh(),  # activation function in the decoder
    "deconv_norms": (None, *("layer" for _ in range(5))),  # normalization in the decoder
    "pools": (2, 2, 2, 2, 2, 2),  # pooling in the encoder
    "rnn_dims": (64, ),  # dimensions of the RNN (First value is not RNN but FC)
    "rnn_bias": True,  # bias in the RNN
    "rnn_dropouts": 0.0,  # dropout in the RNN
    "rnn_funs": torch.nn.GELU(),  # activation function in the RNN
    "n_tasks": 3,  # number of tasks
    "task_layers": 1, # number of layers to use for the tasks (-1 means all layers and 0 means no layers except the bottleneck)
    "task_weight": True,  # use tasks embeddings for the decoder channels (multiplicative)
    "task_bias": True,  # use tasks embeddings for the decoder channels  (additive)
    "task_funs": torch.nn.Tanh(),  # activation function for the tasks embeddings
    "rnn_to_fc": True,  # Whether to use the RNN layers or FC
    'trans_fun': torch.nn.Identity(),  # activation function between Convolutional(.T) and RNN/Linear layers
}

tasks = OrderedDict({})
tasks["Recognition"] = {
    "composer": Broken_CIFAR,
    "key": 0,
    "params": {"in_dims": (3, 64, 64), "out_dims": (3, 192, 192), "n_iter": 1, "gap": 8, "noise": 0.25, "hard": False},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 0.0, 1.0),  # labels, masks, last label
    "loss_s": (None, None),  # labels, masks
    "has_prompt": False,
}
# tasks["Recognition"] = {
#     "composer": Scattered_CIFAR,
#     "key": 0,
#     "params": {"in_dims": (3, 64, 64), "n_iter": 1, "n_grid": 3, "n_pieces": 4, "separate": False, "roll": False},
#     "datasets": [],
#     "dataloaders": [],
#     "loss_w": (0.0, 0.0, 1.0),  # labels, masks, last label
#     "loss_s": (None, None),  # labels, masks
#     "has_prompt": False,
# }
# tasks["Cued_Recognition"] = {
#     "composer": Cued_Scattered_CIFAR,
#     "key": 1,
#     "params": {"in_dims": (3, 64, 64), "fix_attend": (2, 1), "n_grid": 3, "n_pieces": 1, "separate": True, "roll": True, "natural": True, "frame_cue": False},
#     "datasets": [],
#     "dataloaders": [],
#     "loss_w": (0.0, 0.0, 1.0),  # labels, masks, last label
#     "loss_s": (None, None),  # labels, masks
#     "has_prompt": False,
# }
tasks["Cued_Recognition"] = {
    "composer": Cued_CIFAR,
    "key": 1,
    "params": {"in_dims": (3, 64, 64), "fix_attend": (2, 1), "n_grid": 3, "roll": True},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.5, 0.0, 1.0),  # labels, masks, last label
    "loss_s": ((2, ), None),  # labels, masks
    "has_prompt": False,
}
tasks["Search"] = {
    "composer": Search_CIFAR,
    "key": 2,
    "params": {"in_dims": (3, 64, 64), "n_iter": 2, "n_grid": 3, "noise": 0.25, "n_classes": 10},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 10.0, 0.0),  # labels, masks, last label
    "loss_s": (None, slice(0, None)),  # labels, masks
    "has_prompt": True,
}

results_folder, logger = startup_folders(r"./results", name=f"exp_a_{argus.exase}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")

# datasets and dataloaders
pre_trans = transforms.Compose([
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor()
])
train_ds = datasets.STL10(root=data_path, split="train", download=False, transform=pre_trans)
stl_eval_test = datasets.STL10(root=data_path, split="test", download=False, transform=pre_trans)
valid_ds = torch.utils.data.Subset(stl_eval_test, range(4000))
test_ds = torch.utils.data.Subset(stl_eval_test, range(4000, 8000))

DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](train_ds, **tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](valid_ds, **tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](test_ds, **tasks[o]["params"]))
    tasks[o]["datasets"][1].build_valid_test()
    tasks[o]["datasets"][2].build_valid_test()
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory)

# model and optimizer...
model = AttentionModel(**model_params)
(argus.verbose == 1) and logger.info(model)
(argus.verbose == 1) and logger.info(f"Model has {get_n_parameters(model):,} parameters!")
if train_params["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lr"], momentum=0.9, weight_decay=train_params["l2"])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
if train_params["scheduler"] == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_params["n_epochs"], train_params["lr_min"])
elif train_params["scheduler"] == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', train_params["gamma"], patience=8, threshold=1e-2, cooldown=8, min_lr=train_params["lr_min"])
elif train_params["scheduler"] == "OneCycleLR":
    train_params["total_steps"] = len(tasks) * (1 + train_params["n_epochs"]) * (len(train_ds) // train_params["batch_size"])
    train_params["pct_start"] = 0.125
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, train_params["lr"], total_steps=train_params["total_steps"], pct_start=train_params["pct_start"])
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, results_folder, train_params["max_grad_norm"], True, time_it=True)

(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
logger.info(optimizer)
logger.info(scheduler)

# training...
plot_all(10, model, tasks, results_folder, "_train_pre", DeVice, logger, (argus.verbose == 1), kind="train")
plot_all(10, model, tasks, results_folder, "_valid_pre", DeVice, logger, (argus.verbose == 1), kind="valid")
conductor.eval(DeVice)
conductor.train(train_params["n_epochs"], DeVice, True, train_params["mask_mp"], train_params["slow_soft"], train_params["sequential"])
conductor.eval(DeVice)
plot_loss_all(conductor, results_folder)
plot_all(10, model, tasks, results_folder, "_post", DeVice, logger, False)

# saving...
(argus.verbose == 1) and logger.info("Saving results...")
save_dicts(tasks, results_folder, "tasks", logger)
save_dicts(train_params, results_folder, "train_params", logger)
save_dicts(model_params, results_folder, "model_params", logger)
torch.save(torch.tensor(conductor.loss_records), os.path.join(results_folder, "loss_records" + ".pth"))
torch.save(torch.tensor(conductor.valid_records), os.path.join(results_folder, "valid_records" + ".pth"))

torch.save(model.state_dict(), os.path.join(results_folder, "model" + ".pth"))
torch.save(optimizer.state_dict(), os.path.join(results_folder, "optimizer" + ".pth"))
for i, task in enumerate(tasks):
    save_results_to_csv(conductor.loss_records[i], 
                        os.path.join(results_folder, f"loss_{task}.csv"),
                        ["labels", "masks", "last_label"], logger)
    save_results_to_csv(conductor.valid_records[i], 
                        os.path.join(results_folder, f"valid_{task}.csv"),
                        ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)
(argus.verbose == 1) and logger.info("Done!")
