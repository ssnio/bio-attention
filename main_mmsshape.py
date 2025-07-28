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
from src.composer import ShapeSearch_MM, ShapeRecognition_MM
from src.modelv2 import AttentionModel
from src.utils import plot_all, plot_loss_all
from src.utils import build_loaders, get_n_parameters
from src.conductor import AttentionTrain

# # reproducibility
torch.manual_seed(1821)  # Hermann von Helmholtz (1821)
random.seed(1821)


parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=64)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-l2', type=float, default=5e-5)
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
    "milestones": [64, ],
    "gamma": 0.2,
    "mask_mp": 0.0,
    "max_grad_norm": 10.0,
    "scheduler": "CosineAnnealingLR",  # OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR
    "optimizer": "Adam",  # "SGD", "Adam",
    "lr_min": 1e-5,
}

model_params = {
    "channels": [8, 16, 32, 64, 128],
    "fun": torch.nn.GELU(),
    "in_dims": (3, 256, 256),
    "mid_dim": 64,
    "out_dim": 32,
    "n_classes": 18,
    "n_tasks": 2,
    "norm": "layer",
    "softness": 0.5,
    "task_fun": torch.nn.Tanh(),
    "concat": True,
    "skip_maxpool": False,
    "first_k": 3,
}

# # tasks include the composer, key, params, datasets, dataloaders, loss weights, loss slices, and has prompt
# # Loss weights are for the Cross-Entropy (CE), MSE for attention, and CE for the last label prediction
# # the first CE loss is for the iterations indicated in the loss slices
# # the second CE loss is for the last prediction (after applying the last attention map)
# # Loss slices determine which iterations are used for the loss
tasks = OrderedDict({})
tasks["Recognition"] = {
    "composer": ShapeRecognition_MM,  # composer (torch Dataset)
    "key": 0,  # key for the task
    "params": {"n_grid": 4, "n_iter": 1, "directory": data_path},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 0.0, 0.1),  # Loss weights for each modality
    "loss_s": (slice(0, None), None),  # Loss slices (CE, MSE for attention)
    "m_slice": (slice(0, 9), slice(9, 15), slice(15, None)),
    "has_prompt": False,  # has prompt or not (only used for top-down Search)
}
tasks["Search"] = {
    "composer": ShapeSearch_MM,  # composer (torch Dataset)
    "key": 1,  # key for the task
    "params": {"n_grid": 4, "n_iter": 3, "directory": data_path},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 1.0, 0.0),  # Loss weights for each modality
    "loss_s": (None, slice(1, None)),  # Loss slices (CE, MSE for attention)
    "m_slice": (slice(0, 9), slice(9, 15), slice(15, None)),
    "has_prompt": True,  # has prompt or not (only used for top-down Search)
}

results_folder, logger = startup_folders(r"./results", name=f"exp_a_{argus.exase}")
for i, k in enumerate(tasks):
    assert tasks[k]["key"] == i, f"Key {tasks[k]['key']} must be equal to index {i}!"
(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")

# datasets and dataloaders
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"]))
    tasks[o]["datasets"][1].build_valid_test()
    tasks[o]["datasets"][2].build_valid_test()
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory)

# model and optimizer...
model = AttentionModel(**model_params)
(argus.verbose == 1) and logger.info(model)
(argus.verbose == 1) and logger.info(model.map_dims)
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
    train_params["total_steps"] = len(tasks) * (1 + train_params["n_epochs"]) * (len(tasks["Recognition"]["dataloaders"][0].dataset) // train_params["batch_size"])
    train_params["pct_start"] = 0.125
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, train_params["lr"], total_steps=train_params["total_steps"], pct_start=train_params["pct_start"])
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, results_folder, train_params["max_grad_norm"], False)

# training...
plot_all(10, model, tasks, results_folder, "_valid_pre", DeVice, logger, (argus.verbose == 1), kind = "valid")
plot_all(10, model, tasks, results_folder, "_train_pre", DeVice, logger, (argus.verbose == 1), kind = "train")
conductor.eval(DeVice)
conductor.train(train_params["n_epochs"], DeVice, True)
plot_loss_all(conductor, results_folder)
conductor.eval(DeVice)
plot_all(10, model, tasks, results_folder, "_valid_post", DeVice, logger, False, kind = "valid")
plot_all(10, model, tasks, results_folder, "_train_post", DeVice, logger, False, kind = "train")

# saving...
(argus.verbose == 1) and logger.info("Saving results...")
save_dicts(tasks, results_folder, "tasks", logger)
save_dicts(train_params, results_folder, "train_params", logger)
save_dicts(model_params, results_folder, "model_params", logger)
torch.save(model.state_dict(), os.path.join(results_folder, "model" + ".pth"))
torch.save(optimizer.state_dict(), os.path.join(results_folder, "optimizer" + ".pth"))
if hasattr(conductor, "valid_mod_records"):
    if conductor.valid_mod_records is not None:
        torch.save(torch.tensor(conductor.valid_mod_records), os.path.join(results_folder, "valid_mod_records" + ".pth"))
if hasattr(conductor, "train_mod_records"):
    if conductor.train_mod_records is not None:
        torch.save(torch.tensor(conductor.train_mod_records), os.path.join(results_folder, "train_mod_records" + ".pth"))
for i, task in enumerate(tasks):
    save_results_to_csv(conductor.loss_records[i], 
                        os.path.join(results_folder, f"loss_{task}.csv"),
                        ["labels", "masks", "last_label"], logger)
    save_results_to_csv(conductor.valid_records[i], 
                        os.path.join(results_folder, f"valid_{task}.csv"),
                        ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)
(argus.verbose == 1) and logger.info("Done!")
