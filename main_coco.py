import torch
import os
import argparse
from pprint import pformat
from collections import OrderedDict
from src.composer import COCOTokens, COCOAnimals, BG20k
from src.composer import PerceptualGrouping_COCO, Recognition_COCO, Search_COCO, SearchGrid_COCO
from src.conductor import AttentionTrain
from src.model import AttentionModel
from src.utils import plot_all, plot_loss_all
from src.utils import build_loaders, get_n_parameters
from prelude import get_device, startup_folders, save_dicts, save_results_to_csv

parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=42)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-l2', type=float, default=1e-5)
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
    "milestones": [24, 32, ],
    "gamma": 0.2,
    "max_grad_norm": 10.0,
}

model_params = {
    "in_dims": (3, 256, 256),  # input dimensions (channels, height, width)
    "n_classes": 10,  # number of classes
    "out_dim": 20,  # output dimensions (could be larger than n_classes)
    "normalize": True,  # normalize input images
    "softness": [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],  # softness of the attention (scale)
    "channels": (3, 32, 32, 64, 64, 128, 128, 256, 256),  # channels in the encoder
    "residuals": False,  # use residuals in the encoder
    "kernels": (7, 3, 3, 3, 3, 3, 3, 3),  # kernel size
    "strides": 1,  # stride
    "paddings": "same",  # padding
    "conv_bias": True,  # bias in the convolutions
    "conv_norms": (None, *("layer" for _ in range(7))),  # normalization in the encoder
    "conv_dropouts": 0.0,  # dropout in the encoder
    "conv_funs": torch.nn.GELU(),  # activation function in the encoder
    "deconv_funs": torch.nn.GELU(),  # activation function in the decoder
    "deconv_norms": (None, *("layer" for _ in range(7))),  # normalization in the decoder
    "pools": 2,  # pooling in the encoder
    "rnn_dims": (256, ),  # dimensions of the RNN (First value is not RNN but FC)
    "rnn_bias": True,  # bias in the RNN
    "rnn_dropouts": 0.0,  # dropout in the RNN
    "rnn_funs": torch.nn.GELU(),  # activation function in the RNN
    "n_tasks": 3,  # number of tasks
    "task_layers": 1, # number of layers to use for the tasks (-1 means all layers and 0 means no layers except the bottleneck)
    "task_weight": True,  # use tasks embeddings for the decoder channels (multiplicative)
    "task_bias": True,  # use tasks embeddings for the decoder channels  (additive)
    "task_funs": torch.nn.Tanh(),  # activation function for the tasks embeddings
    "rnn_to_fc": False,  # use FC instead of RNN
    'trans_fun': torch.nn.Identity(),  # activation function between Convolutional(.T) and RNN/Linear layers
}

tasks = OrderedDict({})
tasks["Recognition"] = {
    "composer": Recognition_COCO,
    "key": 0,
    "params": {"n_iter": 3, "stride": 64, "blank": False, "static": False, "noise": 0.25},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.1, 0.0, 0.5),  # labels, masks, last label
    "loss_s": (slice(1, None, None), None),  # labels, masks
    "aux_params": None,
    "class_weights": None,
    "has_prompt": False,
    "random": None,
}
tasks["PerceptualGrouping"] = {
    "composer": PerceptualGrouping_COCO,
    "key": 1,
    "params": {"fix_attend": (2, 3), "noise": 0.25},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 1.0, 0.1),  # labels, masks, last label
    "loss_s": (None, slice(1, None, None)),  # labels, masks
    "aux_params": None,
    "class_weights": None,
    "has_prompt": False,
    "random": None,
}
tasks["Search"] = {
    "composer": Search_COCO,
    "key": 2,
    "params": {"n_iter": 3, "noise": 0.25},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 1.0, 0.0),  # labels, masks, last label
    "loss_s": (None, slice(1, None, None)),  # labels, masks
    "aux_params": None,
    "has_prompt": True,
    "random": None,
}
tasks["SearchGrid"] = {
    "composer": SearchGrid_COCO,
    "key": 2,
    "params": {"n_iter": 3, "noise": 0.25},
    "datasets": [],
    "dataloaders": [],
    "loss_w": (0.0, 1.0, 0.0),  # labels, masks, last label
    "loss_s": (None, slice(1, None, None)),  # labels, masks
    "aux_params": None,
    "has_prompt": True,
    "random": None,
}

results_folder, logger = startup_folders(r"./results", name=f"exp_a_coco_{argus.exase}")
(argus.verbose == 1) and logger.info(f"train_params\n {pformat(train_params)}")
(argus.verbose == 1) and logger.info(f"model_params\n {pformat(model_params)}")
(argus.verbose == 1) and logger.info(f"tasks\n {pformat(tasks)}")

# datasets and dataloaders
coco_tokens = COCOTokens(directory=data_path, animals=True, split=0.9)
train_tks, valid_tks, test_tks = coco_tokens.get_tokens()
train_coco = COCOAnimals(in_dims=model_params["in_dims"], directory=data_path, kind=0, tokens=train_tks)
valid_coco = COCOAnimals(in_dims=model_params["in_dims"], directory=data_path, kind=1, tokens=valid_tks)
test_coco = COCOAnimals(in_dims=model_params["in_dims"], directory=data_path, kind=2, tokens=test_tks)
train_bg = BG20k(root=data_path, kind="train")
test_bg = valid_bg = BG20k(root=data_path, kind="test")
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    if tasks[o]["composer"] in (Recognition_COCO , SearchGrid_COCO):
        tasks[o]["datasets"].append(tasks[o]["composer"](train_coco, **tasks[o]["params"], bg_dataset=train_bg))
        tasks[o]["datasets"].append(tasks[o]["composer"](valid_coco, **tasks[o]["params"], bg_dataset=valid_bg))
        tasks[o]["datasets"].append(tasks[o]["composer"](test_coco, **tasks[o]["params"], bg_dataset=test_bg))
    else:
        tasks[o]["datasets"].append(tasks[o]["composer"](train_coco, **tasks[o]["params"]))
        tasks[o]["datasets"].append(tasks[o]["composer"](valid_coco, **tasks[o]["params"]))
        tasks[o]["datasets"].append(tasks[o]["composer"](test_coco, **tasks[o]["params"]))
    tasks[o]["datasets"][1].build_valid_test()
    tasks[o]["datasets"][2].build_valid_test()
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
assert model_params["n_classes"] == train_coco.n_classes, f"Number of n_classes {model_params['n_classes']} and n_classes {train_coco.n_classes} must be equal!"
tasks["PerceptualGrouping"]["class_weights"] = train_coco.class_weights if hasattr(train_coco, "class_weights") else None
tasks["Recognition"]["class_weights"] = train_coco.class_weights if hasattr(train_coco, "class_weights") else None

# model and optimizer...
model = AttentionModel(**model_params)
(argus.verbose == 1) and logger.info(model)
(argus.verbose == 1) and logger.info(f"Model has {get_n_parameters(model):,} parameters!")
optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"], weight_decay=train_params["l2"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params["milestones"], gamma=train_params["gamma"])
conductor = AttentionTrain(model, optimizer, scheduler, tasks, logger, results_folder, train_params["max_grad_norm"], True)

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
                        os.path.join(results_folder, f"eval_{task}.csv"),
                        ["CEi", "CEe", "PixErr", "AttAcc", "ClsAcc"], logger)
(argus.verbose == 1) and logger.info("Done!")
