# # built-in modules
import os
import argparse
from pprint import pformat
from collections import OrderedDict
import random
# # Torch modules
import torch
from torchvision import transforms, datasets
# # internal imports
from prelude import save_dicts, startup_folders, get_device, save_results_to_csv
from src.composer import CelebACrop
from src.model import AttentionModel
from src.utils import plot_all, plot_loss_all
from src.utils import build_loaders, get_n_parameters
from src.conductor import AttentionTrain

# reproducibility
torch.manual_seed(1984)  # Posner & Cohen "Components of visual orienting." (1984)
random.seed(1984)


parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=32)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-l2', type=float, default=1e-4)
parser.add_argument('-lm', type=float, default=0.0)
parser.add_argument('-n_iter', type=int, default=1)
parser.add_argument('-exase', type=str, default="default")
parser.add_argument('-verbose', type=int, default=1)
argus = parser.parse_args()
data_path = r"../attention/data"
train_params = {
    "n_epochs": argus.n_epochs,
    "batch_size": argus.batch_size,
    "lr": argus.lr,
    "l2": argus.l2,
    "lm": argus.lm,
    "exase": argus.exase,
    "dir": r"./results",
    "milestones": [32, ],
    "gamma": 0.2,
}

model_params = {
    "in_dims": (3, 128, 128),  # input dimensions (channels, height, width)
    "n_classes": 2,  # number of classes
    "out_dim": 2,  # output dimensions (could be larger than n_classes)
    "normalize": True,  # normalize input images
    "softness": 0.5,  # softness of the attention (scale)
    "channels": (3, 4, 8, 16, 16, 32),  # channels in the encoder
    "residuals": False,  # use residuals in the encoder
    "kernels": 3,  # kernel size
    "strides": 1,  # stride
    "paddings": "same",  # padding
    "conv_bias": True,  # bias in the convolutions
    "conv_norms": None,  # normalization in the encoder
    "conv_dropouts": 0.0,  # dropout in the encoder
    "conv_funs": torch.nn.ReLU(),  # activation function in the encoder
    "deconv_funs": torch.nn.Tanh(),  # activation function in the decoder
    "deconv_norms": None,  # normalization in the decoder
    "pools": (4, 2, 2, 2, 2),  # pooling in the encoder
    "rnn_dims": (16, 8),  # dimensions of the RNN (First value is not RNN but FC)
    "rnn_bias": True,  # bias in the RNN
    "rnn_dropouts": 0.0,  # dropout in the RNN
    "rnn_funs": torch.nn.ReLU(),  # activation function in the RNN
    "n_tasks": 1,  # number of tasks
    "task_weight": False,  # use tasks embeddings for the decoder channels (multiplicative)
    "task_bias": False,  # use tasks embeddings for the decoder channels  (additive)
    "task_funs": None,  # activation function for the tasks embeddings
    "rnn_to_fc": True,  # Whether to use the RNN layers or FC
    "rnn_cat": False, # whether to concatenate the forward and backward RNN outputs
    "use_bridges": False,  # whether to use a fancy bridge between the encoder and decoder
}

# # tasks include the composer, key, params, datasets, dataloaders, loss weights, loss slices, and has prompt
# # Loss weights are for the Cross-Entropy (CE), MSE for attention, and CE for the last label
# # the first CE loss is for the iterations indicated in the loss slices
# # the second CE loss is for the last label (after applying the last attention map)
# # Loss slices determine which iterations are used for the loss
tasks = OrderedDict({})
tasks["Celeb"] = {
    "composer": CelebACrop,  # composer (torch Dataset)
    "key": 0,  # key for the task
    "params": {"n_iter": argus.n_iter, "hair_dir": None, "in_dims": model_params["in_dims"], "padding": 0, "noise": 0.125, "which": 0},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 0.0, 1.0),  # Loss weights (Cross-Entropy (CE), MSE for attention, CE last label)
    "loss_s": (None, None),  # Loss slices (CE, MSE for attention)
    "has_prompt": False,  # has prompt or not (only used for top-down Search)
}

model_params["n_tasks"] = len(tasks)
results_folder, logger = startup_folders(r"./results", name=f"exp_a_{argus.exase}")
for i, k in enumerate(tasks):
    assert tasks[k]["key"] == i, f"Key {tasks[k]['key']} must be equal to index {i}!"
(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")

# datasets and dataloaders
train_ds = datasets.CelebA(root=data_path, split='train', transform=transforms.ToTensor())
valid_ds = datasets.CelebA(root=data_path, split='valid', transform=transforms.ToTensor())
test_ds = datasets.CelebA(root=data_path, split='test', transform=transforms.ToTensor())
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](train_ds, **tasks[o]["params"], kind="train"))
    tasks[o]["datasets"].append(tasks[o]["composer"](valid_ds, **tasks[o]["params"], kind="valid"))
    tasks[o]["datasets"].append(tasks[o]["composer"](test_ds, **tasks[o]["params"], kind="test"))
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
conductor.train(train_params["n_epochs"], DeVice, True, train_params["lm"])
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
    save_results_to_csv(conductor.eval_records[i], 
                        os.path.join(results_folder, f"eval_{task}.csv"),
                        ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)
(argus.verbose == 1) and logger.info("Done!")
