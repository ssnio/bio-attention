# # built-in modules
import os
import argparse
from pprint import pformat
from collections import OrderedDict
import random
# # Torch modules
import torch
# # internal imports
from prelude import save_dicts, startup_folders, get_device, save_results_to_csv
from src.composer import CurveTracing
from src.model import AttentionModel
from src.utils import plot_all, plot_loss_all
from src.utils import build_loaders, get_n_parameters
from src.conductor import AttentionTrain

# # reproducibility
torch.manual_seed(1998)  # Roelfsema et al., Object-based attention (1998)
random.seed(1998)

parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-l2', type=float, default=1e-6)
parser.add_argument('-mode', type=int, default=0)  # 0: curve-supervision, 1: no-curve-supervision
parser.add_argument('-exase', type=str, default="default")
parser.add_argument('-verbose', type=int, default=1)
argus = parser.parse_args()

train_params = {
    "n_epochs": argus.n_epochs,
    "batch_size": argus.batch_size,
    "lr": argus.lr,
    "l2": argus.l2,
    "exase": argus.exase,
    "dir": r"./results",
    "milestones": [128, ],
    "gamma": 0.1,
    "mode": argus.mode,
}

model_params = {
    "in_dims": (1, 128, 128),  # input dimensions (channels, height, width)
    "n_classes": 16,  # number of classes
    "out_dim": 16,  # output dimensions (could be larger than n_classes)
    "normalize": False,  # normalize input images
    "softness": 0.5,  # softness of the attention (scale)
    "channels": (1, 4, 4, 8, 8, 16, 16, 32, 32),  # channels in the encoder
    "residuals": False,  # use residuals in the encoder
    "kernels": 3,  # kernel size
    "strides": 1,  # stride
    "paddings": 1,  # padding
    "conv_bias": False,  # bias in the convolutions
    "conv_norms": None,  # normalization in the encoder
    "conv_dropouts": 0.1,  # dropout in the encoder
    "conv_funs": torch.nn.ReLU(),  # activation function in the encoder
    "deconv_funs": torch.nn.Tanh(),  # activation function in the decoder
    "deconv_norms": (None, *("layer" for _ in range(7))),  # normalization in the decoder
    "pools": (1, 2, 1, 2, 1, 2, 1, 2),  # pooling in the encoder
    "rnn_dims": (128, ),  # dimensions of the RNN (First value is not RNN but FC)
    "rnn_bias": True,  # bias in the RNN
    "rnn_dropouts": 0.0,  # dropout in the RNN
    "rnn_funs": torch.nn.ReLU(),  # activation function in the RNN
    "n_tasks": 1,  # number of tasks
    "task_weight": False,  # use tasks embeddings for the decoder channels (multiplicative)
    "task_bias": False,  # use tasks embeddings for the decoder channels  (additive)
    "task_funs": None,  # activation function for the tasks embeddings
    "rnn_to_fc": False,  # Whether to use the RNN layers or FC
    "trans_fun": torch.nn.Identity(),
}

if train_params["mode"] == 0:
    super_loss = (1, 3, 5)
elif train_params["mode"] == 1:
    super_loss = (0, 1, 4, 5)

tasks = OrderedDict({})
tasks["CurveTracing"] = {
    "composer": CurveTracing,
    "key": 0,
    "params": {"fix_attend_saccade": (2, 3, 2), "ydim": 96, "xdim": 96, "padding": 16, "resolution": 100, "noise": 0.25},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 1.0, 0.0),  # labels, masks, last label
    "loss_s": (None, super_loss),  # labels, masks
    "has_prompt": False,
}

model_params["n_tasks"] = len(tasks)
results_folder, logger = startup_folders(r"./results", name=f"exp_a_{argus.exase}")
for i, k in enumerate(tasks):
    assert tasks[k]["key"] == i, f"Key {tasks[k]['key']} must be equal to index {i}!"
(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")

# datasets and dataloaders
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**14, **tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**10, **tasks[o]["params"]))
    tasks[o]["datasets"][-1].build_valid_test()
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**10, **tasks[o]["params"]))
    tasks[o]["datasets"][-1].build_valid_test()
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory)

# model and optimizer...
model = AttentionModel(**model_params)
(argus.verbose == 1) and logger.info(model)
(argus.verbose == 1) and logger.info(f"Model has {get_n_parameters(model):,} parameters!")
optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, results_folder)

# training...
plot_all(10, model, tasks, results_folder, "_pre", DeVice, logger, (argus.verbose == 1))
conductor.eval(DeVice)
conductor.train(train_params["n_epochs"], DeVice, True)
plot_loss_all(conductor, results_folder)
conductor.eval(DeVice)
plot_all(10, model, tasks, results_folder, "_post", DeVice, logger, False)

# saving...
(argus.verbose == 1) and logger.info("Saving results...")
save_dicts(tasks, results_folder, "tasks", logger)
save_dicts(train_params, results_folder, "train_params", logger)
save_dicts(model_params, results_folder, "model_params", logger)
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
