import torch
import os
import argparse
from pprint import pformat
from collections import OrderedDict
from src.model import AttentionModel
from src.composer import PsychChange
from src.conductor import seq_cls_train, seq_cls_eval
from src.utils import plot_all
from src.utils import build_loaders, get_n_parameters
from prelude import get_device, startup_folders, save_dicts
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-n_epochs", type=int, default=64)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=0.0005)
parser.add_argument("-l2", type=float, default=0.0001)
parser.add_argument("-bias", type=float, default=0.5)
parser.add_argument("-nobj", type=int, default=-1)
parser.add_argument("-exase", type=str, default="default")
parser.add_argument("-verbose", type=int, default=1)
argus = parser.parse_args()
results_folder, logger = startup_folders(r"./results", name=f"exp_psycphys_{argus.exase}")

train_params = {
    "n_epochs": argus.n_epochs,
    "batch_size": argus.batch_size,
    "lr": argus.lr,
    "l2": argus.l2,
    "exase": argus.exase,
    "dir": r"./results",
    "milestones": [32, 64, ],
    "gamma": 0.2,
    "max_grad_norm": 10.0,
    "lr_min": 1e-5,
    "scheduler": "OneCycleLR",  # OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR
    "optimizer": "Adam",  # "SGD", "Adam",
    "n_samples": 8*1024,  # number of samples per task
}

model_params = {
    "in_dims": (1, 96, 96),  # input dimensions (channels, height, width)
    "n_classes": 9,  # number of classes
    "out_dim": 9,  # output dimensions (could be larger than n_classes)
    "normalize": True,  # normalize input images
    "softness": 0.5,  # softness of the attention (scale)
    "channels": (1, 4, 8, 16, 32, 32),  # channels in the encoder
    "residuals": False,  # use residuals in the encoder
    "kernels": 3,  # kernel size
    "strides": 2,  # stride
    "paddings": 1,  # padding
    "conv_bias": True,  # bias in the convolutions
    "conv_norms": (None, *("layer" for _ in range(4))),  # normalization in the encoder
    "conv_dropouts": 0.0,  # dropout in the encoder
    "conv_funs": torch.nn.ReLU(),  # activation function in the encoder
    "deconv_funs": torch.nn.Tanh(),  # activation function in the decoder
    "deconv_norms": (None, *("layer" for _ in range(4))),  # normalization in the decoder
    "pools": (1, 1, 1, 1, 1),  # pooling in the encoder
    "rnn_dims": (16, 16),  # dimensions of the RNN (First value is not RNN but FC)
    "rnn_bias": True,  # bias in the RNN
    "rnn_dropouts": 0.0,  # dropout in the RNN
    "rnn_funs": torch.nn.ReLU(),  # activation function in the RNN
    "n_tasks": 1,  # number of tasks
    "task_layers": 1, # number of layers to use for the tasks (-1 means all layers and 0 means no layers except the bottleneck)
    "task_weight": True,  # use tasks embeddings for the decoder channels (multiplicative)
    "task_bias": True,  # use tasks embeddings for the decoder channels  (additive)
    "task_funs": torch.nn.Tanh(),  # activation function for the tasks embeddings
    'norm_mean': [0.5, ],  # mean for the normalization
    'norm_std': [0.25, ],  # std for the normalization
    "rnn_to_fc": False,  # Whether to use the RNN layers or FC
    'trans_fun': torch.nn.Identity(),  # activation function between Convolutional(.T) and RNN/Linear layers
}

tasks = OrderedDict({})
tasks["RotationChangeDetection"] = {
    "composer": PsychChange,
    "key": 0,
    "params": {
        "episodes": (0, 2, 0, 3),
        "r_range": 30,
        "r_base": 5,
        "biased": argus.bias,
        "force_range": False,
        "force_label": False,
        "n_objects": argus.nobj,
        "noise": 0.25,
    },
    "datasets": [],
    "dataloaders": [],
    "loss_w": ((1.0, 1.0), (1.0, 1.0)),  # labels, masks
    "loss_s": ((-2,   -1), (0,   1)),  # labels, masks
}

# datasets and dataloaders
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"], n_samples=train_params["n_samples"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"], n_samples=1024))
    tasks[o]["datasets"].append(tasks[o]["composer"](**tasks[o]["params"], n_samples=1024))
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

# model and optimizer...
model = AttentionModel(**model_params)
optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
if train_params["scheduler"] == "OneCycleLR":
    train_params["total_steps"] = (1 + train_params["n_epochs"]) * (train_params["n_samples"] // train_params["batch_size"])
    train_params["pct_start"] = 0.25
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, train_params["lr"], total_steps=train_params["total_steps"], pct_start=train_params["pct_start"])
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])

(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")
(argus.verbose == 1) and logger.info(f"Model has {get_n_parameters(model):,} parameters!")
(argus.verbose == 1) and logger.info(model)

# training...
plot_all(10, model, tasks, results_folder, "_pre", DeVice, logger, (argus.verbose == 1))
_ = seq_cls_eval(model, tasks, DeVice, logger, valid=True, verbose=True)
loss_log, eval_log = seq_cls_train(model, tasks, optimizer, scheduler, train_params["n_epochs"], DeVice, logger, train_params["max_grad_norm"])
_ = seq_cls_eval(model, tasks, DeVice, logger, valid=True, verbose=True)
plot_all(10, model, tasks, results_folder, "_post", DeVice, logger, False)

plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
plt.plot(loss_log, label="loss_log")
plt.legend()
plt.subplot(3, 1, 2)
for i in range(len(eval_log[0])):
    plt.plot(eval_log[0][i], label=f"{i} CE")
plt.legend()
plt.subplot(3, 1, 3)
for i in range(len(eval_log[0])):
    plt.plot(eval_log[1][i], label=f"{i} Acc")
plt.legend()
plt.savefig(os.path.join(results_folder, "loss.svg"), format='svg')
plt.close()

# saving...
(argus.verbose == 1) and logger.info("Saving results...")
save_dicts(train_params, results_folder, "train_params", logger)
save_dicts(model_params, results_folder, "model_params", logger)
save_dicts(tasks, results_folder, "tasks", logger)
torch.save(model.state_dict(), os.path.join(results_folder, "model" + ".pth"))
torch.save(optimizer.state_dict(), os.path.join(results_folder, "optimizer" + ".pth"))
torch.save(torch.tensor(loss_log), os.path.join(results_folder, "loss_log.pth"))
eval_names = ["ce_loss", "accuracy", "mask_loss"]
for i, s in enumerate(eval_log):
    if len(s) > 0:
        torch.save(torch.tensor(s), os.path.join(results_folder, f"eval_{eval_names[i]}_log.pth"))
(argus.verbose == 1) and logger.info("Done!")
