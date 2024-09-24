# # built-in modules
from typing import Any
# # Torch modules
import torch
from torch.utils.data import Dataset, DataLoader
# # other modules
import matplotlib.pyplot as plt


def obj_to_tuple(obj: Any, n: int, do: bool = False):
    # if the object is not iterable, return a tuple of the object repeated n times
    # if the do flag is set to True, returns a tuple of the object repeated n times
    # else return the object as is
    return obj if (isinstance(obj, (tuple, list, set)) and not do) else tuple(obj for _ in range(n))


def plot_one(n, model_, dataloader_, key_, has_prompt_, directory, logger, prefix: str = None, suffix: str = None, device="cpu", verbose=False):
    prefix = prefix if prefix is not None else ""
    suffix = suffix if suffix is not None else ""
    composites, labels, masks, components, hot_labels = next(iter(dataloader_))
    verbose and logger.info(f"  composites: {tuple(composites.shape)}")
    verbose and logger.info(f"  labels: {tuple(labels.shape)}",)
    verbose and logger.info(f"  masks: {tuple(masks.shape)}")
    verbose and logger.info(f"  components: {tuple(components.shape)}")
    verbose and logger.info(f"  hot_labels: {tuple(hot_labels.shape)}")
    with torch.no_grad():
        model_.eval()
        model_.to(device)
        composites = composites.to(device)
        hot_labels = hot_labels.to(device) if has_prompt_ else None
        p_masks, *_ = model_(composites, key_, hot_labels if has_prompt_ else None)
        composites = composites.cpu()
        p_masks = p_masks.cpu()

    im_size = 2
    n_iter = composites.size(1)
    n = min(n, composites.size(0))
    plt.figure(figsize=(3 * n * im_size, n_iter * im_size))
    for j in range(n):
        for i in range(n_iter):
            plt.subplot(n_iter, 3 * n, 1 + i * 3 * n + j * 3)
            plt.imshow(composites[j, i].permute(1, 2, 0))
            plt.axis("off")
        for i in range(n_iter):
            plt.subplot(n_iter, 3 * n, 2 + i * 3 * n + j * 3)
            plt.imshow(p_masks[j, i][0], cmap="plasma", vmin=-1, vmax=1)
            plt.axis("off")
        if masks.ndim > 1:
            for i in range(masks.size(1)):
                plt.subplot(n_iter, 3 * n, 3 + i * 3 * n + j * 3)
                plt.imshow(masks[j, i][0], cmap="plasma", vmin=-1, vmax=1)
                plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{directory}/{prefix}{suffix}.svg", format="svg")
    plt.close()


def plot_all(n, model_, tasks_, directory, suffix, device, logger, verbose=False, kind="valid"):

    for task in tasks_:
        verbose and logger.info(f"Plotting {task}...")
        if kind == "train":
            dataloader_ = tasks_[task]["dataloaders"][0]
        elif kind == "valid":
            dataloader_ = tasks_[task]["dataloaders"][1]
        elif kind == "test":
            dataloader_ = tasks_[task]["dataloaders"][2]
        key_ = tasks_[task]["key"]
        has_prompt_ = tasks_[task].get("has_prompt", False)
        plot_one(n, model_, dataloader_, key_, has_prompt_, directory, logger, task, suffix, device, verbose)


def plot_loss(loss_records, directory, prefix: str = None, suffix: str = None):
    prefix = prefix if prefix is not None else ""
    suffix = suffix if suffix is not None else ""
    plt.figure(figsize=(9, 9))
    for i, k in enumerate(["labels", "masks", "last label"]):
        plt.subplot(3, 1, i + 1)
        if isinstance(loss_records[i][0], (int, float)):
            plt.plot(loss_records[i], label=k)
        elif isinstance(loss_records[i][0], (list, tuple)):
            for j in range(len(loss_records[i])):
                plt.plot(loss_records[i][j], c=['r', 'g', 'b'][j], label=f"{k} {['s', 'c', 'p'][j]}")
        else:
            raise ValueError(f"Unknown type {type(loss_records[i][0])}!")
        plt.yscale("log")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{directory}/{prefix}{suffix}.svg", format="svg")
    plt.close()


def plot_loss_all(conductor_, directory, suffix=""):
    loss_records = conductor_.loss_records
    k_tasks = conductor_.k_tasks
    for i, task in enumerate(k_tasks):
        plot_loss(loss_records[i], directory, f"loss_{suffix}", task)


def build_loaders(datasets: Dataset, batch_size: int, num_workers: int = 0, pin_memory: bool = False, shuffle: bool = False):
    train_ds, val_ds, test_ds = datasets
    batch_size = batch_size
    num_workers = num_workers
    pin_memory = pin_memory
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)
    valid_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle)
    return train_loader, valid_loader, test_loader


def get_n_parameters(model: torch.nn.Module):
    i = 0
    for par in model.parameters():
        i += par.numel()
    return i


def get_grad_norms(model: torch.nn.Module):
    grads = []
    for param in model.parameters():
        try:
            grads.append(param.grad.data.cpu().detach().clone().ravel())
        except AttributeError:
            pass
    return torch.linalg.norm(torch.cat(grads))


def get_ior_match(one_: torch.Tensor, all_: torch.Tensor):
    """returns the index of the most similar element in all_ to one_
    """
    return (all_ * one_).sum(dim=(-2, -1)).max(dim=1).indices[:, 0]


def get_dims(x: tuple, layer: torch.nn.Module):
    """ unnecessarily complicated way to
    calculate the output height and width size for a Conv2D/or/MaxPool2d

    Args:
        x (tuple): input size
        layer (nn.Conv2d, nn.MaxPool2d): the Conv2D/or/MaxPool2d layer

    returns:
        (int): output shape as given in
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    p = layer.padding if isinstance(layer.padding, (tuple, list)) else (layer.padding,)
    k = layer.kernel_size if isinstance(layer.kernel_size, (tuple, list)) else (layer.kernel_size,)
    s = layer.stride if isinstance(layer.stride, (tuple, list)) else (layer.stride,)
    x = x if isinstance(x, (tuple, list)) else [x]
    x = x[-2:] if len(x) > 2 else x
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
        d = layer.dilation if isinstance(layer.dilation, (tuple, list)) else (layer.dilation,)
        return (1 + (x[0] + 2 * p[0] - (k[0] - 1) * d[0] - 1) // s[0],
                1 + (x[-1] + 2 * p[-1] - (k[-1] - 1) * d[-1] - 1) // s[-1])
    elif isinstance(layer, torch.nn.AvgPool2d):
        return (1 + (x[0] + 2 * p[0] - k[0]) // s[0],
                1 + (x[-1] + 2 * p[-1] - k[-1]) // s[-1])
