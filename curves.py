# # built-in modules
import os
import argparse
from pprint import pformat
from collections import OrderedDict
import random
# # Torch modules
import torch
# # internal imports
from prelude import save_dicts, startup_folders, get_device, save_results_to_csv, load_dicts
from src.composer import SwitchBox, transforms
from src.model import AttentionModel
from src.utils import plot_all, plot_loss_all, DataLoader
from src.utils import build_loaders, get_n_parameters
from src.conductor import AttentionTrain

import matplotlib
import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
c = matplotlib.colormaps["tab10"]


def gaussian(x, mu, std):
    """
    Gaussian normalized function
    """
    y = torch.exp( - ((x - mu) / std)**2 / 2 ) / (std * math.sqrt(2 * math.pi))
    return y

def roll_to_mean(x):
    d = x.size(-1)
    argmean = torch.argmax(x, dim=-1).item()
    roll = -argmean + d//2 + d%2 - 1
    return torch.roll(x, roll, dims=-1), roll

def try_gaussian_fit(x):
    d = x.numel()
    v = torch.arange(d)
    mean = torch.argmax(x).item()
    std = torch.argmin((torch.cumsum(x, dim=0) - 0.859 * x.sum()).square()).item() - mean
    return gaussian(v, mean, std)

def does_it_fit(x):
    y = try_gaussian_fit(x)
    x = x / torch.linalg.norm(x)
    y = y / torch.linalg.norm(y)
    z = torch.dot(y, x)
    return z

def parameters_of_orientation_tuning(x, k=180):
    a = torch.max(x).item()
    d = torch.argmax(x).item()
    b = (torch.argmin((torch.cumsum(x, dim=0) - 0.859 * x.sum()).square()).item() - d)/k
    c = torch.min(x).item()
    return a, b, c, d

def scale_(x: torch.Tensor, dim=((-2, -1)), eps=1e-6):
    return x / (x.abs().flatten(start_dim=dim[0], end_dim=dim[1]).max(dim=-1, keepdim=True).values + eps)


def gabor(h: float, w: float, sigma: float, theta: float, k: float):
    pi = torch.pi
    rh, rw = int(h//2), int(w//2)
    xx, yy = torch.meshgrid(torch.arange(-rh, rh), torch.arange(-rw, rw), indexing='ij')
    cc = torch.zeros(h, w)

    x =  xx * math.cos(theta) + yy * math.sin(theta)
    y = -xx * math.sin(theta) + yy * math.cos(theta)
    
    gaussian = torch.exp(- sigma**2 / (8 * k**2) * (4 * x**2 + y**2)) * sigma**2 / (4*pi * k**2)
    sinusoid = torch.cos(sigma * x) * math.exp(k**2 / 2)
    stimulus = torch.clamp(scale_(gaussian * sinusoid), 0.0, 1.0)
    cc[rh-1:rh+1, rw-1:rh+1] = 1.0
    return stimulus.unsqueeze(0), cc.unsqueeze(0)

def make_bar(h: float, w: float, theta: float, k = 19, sigma: float = 3.0):
    if k > 0 and sigma > 0:
        blur = transforms.GaussianBlur(k, sigma=(sigma, sigma))
    else:
        blur = lambda x: x
    p = min(h//2, w//2)
    pp = p//4
    x = torch.zeros(1, h, w)
    c = torch.zeros(1, h, w)
    x[:, p-2:p+2, pp:3*pp+p] = 1.0
    c[:, p-2:p+2, pp-2:pp+2] = 1.0
    x = transforms.functional.rotate(x, theta, interpolation=transforms.InterpolationMode.BILINEAR)
    c = transforms.functional.rotate(c, theta, interpolation=transforms.InterpolationMode.BILINEAR)
    x = blur(x)
    c = blur(c)
    x = scale_(x, dim=(-2, -1))
    c = scale_(c, dim=(-2, -1))
    return x, c

def make_stimuli(h: int, w: int, b: int, bar: bool = False):
    assert h%4 == 0 and w%4 == 0
    x  = torch.zeros(b, 1, h, w)
    xc = torch.zeros(b, 1, h, w)
    zc = torch.zeros(b, 1, h, w)
    t = torch.zeros(b, 1, h, w)
    theta = torch.linspace(0., 179., b)
    for i in range(b):
        if bar:
            xx, cc = make_bar(h//2, w//2, theta[i].item())
        else:
            xx, cc = gabor(h//2, w//2, torch.pi/4, theta[i].item()*torch.pi/180.0, 2.0)
        x[i, :, :h//2, :w//2] = xx
        x[i, :, -h//2:, -w//2:] = xx
        xc[i, :, :h//2, :w//2] = cc
        zc[i, :, -h//2:, -w//2:] = cc
        t[i, :, :h//2, :w//2] = xx
    return x, xc, zc, t

def parameters_of_orientation_tuning(x, n=180):
    a, d = x.max(dim=-1)
    b = (torch.argmin((torch.cumsum(x, dim=-1) - 0.859 * x.sum(dim=-1, keepdim=True)).square(), dim=-1) - d)/n
    c = torch.min(x, dim=-1).values
    mu = x.mean(dim=-1)
    md = x.median(dim=-1).values
    return a, b, c, d, mu, md


def modulation_index(a: torch.Tensor, u: torch.Tensor):
    """
    attentional modulation index: (attended - unattended)/(attended + unattended)
    ref: McAdams and Maunsell · Effects of Attention on Orientation Tuning
    """
    return torch.nan_to_num((a - u) / (a + u), nan=0.0, posinf=0.0, neginf=0.0)


def normed_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean', donorm: bool = True) -> float:
    if donorm:
        predictions = (predictions + 1.0) / 2.0
        targets = (targets + 1.0) / 2.0
    corrects = (predictions - targets).square().sum()
    return corrects / targets.numel() if reduction == 'mean' else corrects * targets.size(0) / targets.numel()


def roelf_accuracy(y_t: torch.Tensor, y_d: torch.Tensor, y_p: torch.Tensor):
    y_tp = (y_t * y_p).sum(dim=(-3, -2, -1))
    y_dp = (y_d * y_p).sum(dim=(-3, -2, -1))
    return (y_tp > y_dp).sum().item() / y_t.size(0)

def get_rec_field_act(model: AttentionModel, x: torch.Tensor, e: float = 1e-3):
    # x is the stimulus or receptive field as (batch, channel, h, w)
    # get the receptive field
    list_ind = [[] for _ in range(model.n_convs)]
    base_act = [[] for _ in range(model.n_convs)]
    for i in range(model.n_convs):
        conv = model.conv_blocks[i].conv
        pool = model.conv_blocks[i].pool if model.conv_blocks[i].pool is not None else lambda x: x
        fun = model.conv_blocks[i].fun
        x = pool(fun(conv(x)))
        base_act[i] = x
        list_ind[i] = (x.abs() > e)
    return list_ind, base_act

def get_activity(model: AttentionModel, x: torch.Tensor):
    if x.ndim == 4:  # (batch, channel, h, w)
        x = x.unsqueeze(1)  # (batch, n_iter, channel, h, w)
    batch_size, n_iter = x.size(0), x.size(1)
    all_acts = []
    model.eval()
    with torch.no_grad():
        model.initiate_forward(batch_size=batch_size)
        for i in range(n_iter):
            *_, act_ = model.one_forward(x[:, i])
            all_acts.append(act_)
    return all_acts

def polar_plot(xt, xd, theta_res, results_folder, plotname):
    theta = torch.linspace(0, 2*torch.pi, 2*theta_res)
    fig, ax = plt.subplots(figsize=(3, 2), subplot_kw={'projection': 'polar'})
    r_max = max(xt.max().item(), xd.max().item())
    ax.plot(theta, torch.cat([xt, xt])/r_max, c='#E91EF9')
    ax.plot(theta, torch.cat([xd, xd])/r_max, c='#07BAFC')
    ax.set_rmax(1.05)
    # ax.set_rticks([0.25, 0.5, 0.75])  # Less radial ticks
    # ax.set_xticks(torch.linspace(0, 2*torch.pi, 8))  # Less radial ticks
    # ax.set_xticklabels(["$0^\circ$", None, "$90^\circ$", None, "$180^\circ$", None, "$270^\circ$", None])  # Less radial ticks
    # ax.set_xticklabels([0, 90, 180, 270])  # Less radial ticks
    # ax.set_yticklabels([])  # Less radial ticks
    # ax.set_rlabel_position(0.0)  # Move radial labels away from plotted line
    ax.grid(True)
    # ax.set_title(f"Scaled response of Cell {c} Layer {i} ", va='bottom')
    # if argus.res_fold is not None:
    plt.savefig(os.path.join(results_folder, 'Tuning_Curve_' + plotname + '.svg'))
    plt.close()

def plot_curves(n_layers, curve_tar_act, curve_dis_act, results_folder = None, plotname = None):
    for j in range(n_layers):
        plt.figure(figsize=(6, 4))
        # mean = torch.cat([mean_tar_act[j][:2], mean_dis_act[j][:2]]).mean().cpu()
        plt.plot(curve_tar_act[j].detach().cpu(), color="r")
        plt.plot(curve_dis_act[j].detach().cpu(), color="b")
        plt.title(f"Layer {j}")
        if results_folder is None or plotname is None:
            plt.show()
        else:
            plt.savefig(os.path.join(results_folder, f"{plotname}_{j}.svg"), format="svg")
            plt.close()

# start_folder = r"/Users/saeedida/GitProjects/attention/results/_roelf/new_era/1721344744__cuz"
start_folder = r"/Users/saeedida/GitProjects/attention/results/_new_era/bio/curve/1726912036_sw_c"
results_folder, logger = startup_folders(start_folder, name=f"exp_a")
data_path = r"./data"

model_params = load_dicts(start_folder, "model_params")
tasks = load_dicts(start_folder, "tasks")
train_params = load_dicts(start_folder, "train_params")
DeVice, num_workers, pin_memory = get_device()
print(f"model_params: {model_params}")
print(f"tasks: {tasks}")
print(f"train_params: {train_params}")

# # setting up the tasks
tasks['SwitchBox']["composer"] = SwitchBox
tasks['SwitchBox']["datasets"] = []
tasks['SwitchBox']["dataloaders"] = []
tasks['SwitchBox']["loss_s"] =  [None, [1, 4, 6]]

# datasets and dataloaders
DeVice, num_workers, pin_memory = get_device()
for o in tasks:
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**14, **tasks[o]["params"]))
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**10, **tasks[o]["params"]))
    tasks[o]["datasets"][-1].build_valid_test()
    tasks[o]["datasets"].append(tasks[o]["composer"](n_samples=2**10, **tasks[o]["params"]))
    tasks[o]["datasets"][-1].build_valid_test()
    tasks[o]["dataloaders"] = build_loaders(tasks[o]["datasets"], batch_size=train_params["batch_size"], num_workers=num_workers, pin_memory=pin_memory)

# create a blank model
model = AttentionModel(**model_params)
conductor = AttentionTrain(model, None, None, tasks, logger, results_folder)

# load states into the model
model_dir = os.path.join(start_folder, "model" + ".pth")
assert os.path.exists(model_dir), "Could not find the model.pth in the given dir!"
model.load_state_dict(torch.load(model_dir, map_location=DeVice))

# evaluating...
conductor.eval(DeVice, False)


class Roelfsema:
    def __init__(self, model, tasks):
        import matplotlib.pyplot as plt
        self.model = model
        self.tasks = tasks
        self._loader = tasks['SwitchBox']["dataloaders"][-1]
        self._loader.dataset.training = False
        self.n_samples = len(self._loader.dataset)

        fix_attend_saccade = self.tasks['SwitchBox']["params"]["fix_attend_saccade"]
        n_iter = sum(fix_attend_saccade)
        n_layers = model.n_convs
        n_fix, n_att, n_sac = fix_attend_saccade
        n_fix_att = n_fix + n_att

    def test_accuracy_curve(self, device):
        correct = 0
        threshold = 0.25
        self._loader = tasks['SwitchBox']["dataloaders"][-1]
        self._loader.dataset.training = False
        
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for x, _, m, _, c in self._loader:
                    b = x.size(0)
                    x, m, c = x.to(device), m.to(device), c.to(device)
                    p_m, _, _ = model(x, tasks['SwitchBox']["key"])
                    m_correct = (p_m[:, -1] * c[:, 2])
                    m_false = (p_m[:, -1] * c[:, 5])
                    m_correct = m_correct.view(b, -1).sum(dim=-1) / c[:, 2].view(b, -1).sum(dim=-1)
                    m_false = m_false.view(b, -1).sum(dim=-1) / c[:, 5].view(b, -1).sum(dim=-1)
                    correct += ((m_correct - m_false) > threshold).sum().item()
        print(f"Accuracy: {correct / self.n_samples}")
        self._loader.dataset.training = True


roelfsema_ = Roelfsema(model, tasks)
roelfsema_.test_accuracy_curve(DeVice)



batch_size = 256
tasks["SwitchBox"]["datasets"][-1].training = False
this_dl = DataLoader(tasks["SwitchBox"]["datasets"][-1], batch_size=batch_size, shuffle=False)

target_composites, distractor_composites, masks, rec_fields, components = next(iter(this_dl))
target_composites = target_composites.to(DeVice)
distractor_composites = distractor_composites.to(DeVice)
masks = masks.to(DeVice)
rec_fields = rec_fields.to(DeVice)
components = components.to(DeVice)
both_composites = (components[:, 1:2] + components[:, 4:5]).clamp(0.0, 1.0)

print(target_composites.shape, distractor_composites.shape, masks.shape, rec_fields.shape, components.shape, both_composites.shape)

# get the receptive field
model.eval()
with torch.no_grad():
    # list_ind, base_act = get_rec_field_act(model, rec_fields)
    
    targets_ = get_activity(model, target_composites)
    distractors_ = get_activity(model, distractor_composites)
    tar_cue_ = get_activity(model, components[:, 0:1])
    dis_cue_ = get_activity(model, components[:, 3:4])
    both_ = get_activity(model, both_composites)[0]
    
    model.initiate_forward(batch_size=rec_fields.size(0))
    *_, receptive_ = model.for_forward(rec_fields)
    
    tmasks_, *_ = model(target_composites)
    dmasks_, *_ = model(distractor_composites)


fix_attend_saccade = tasks['SwitchBox']["params"]["fix_attend_saccade"]
n_iter = sum(fix_attend_saccade)
n_layers = model.n_convs
n_fix, n_att, n_sac = fix_attend_saccade
n_fix_att = n_fix + n_att
n_layers = model.n_convs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
q = torch.linspace(0, 0.99, 100)
for i in range(n_iter):
    for j in range(n_layers):
        tar_q = torch.quantile(targets_[i][j][receptive_[j] > 0.0].ravel(), q.to(DeVice))
        dis_q = torch.quantile(distractors_[i][j][receptive_[j] > 0.0].ravel(), q.to(DeVice))
        mean_tar_q = torch.quantile(targets_[i][j][receptive_[j] > 0.0].ravel(), 0.5).cpu()
        mean_dis_q = torch.quantile(distractors_[i][j][receptive_[j] > 0.0].ravel(), 0.5).cpu()
        plt.figure(figsize=(6, 4))
        plt.title(f"Layer {j}, Iteration {i}, {mean_tar_q:.2f}, {mean_dis_q:.2f}")
        plt.plot(tar_q.cpu(), 100.0*q, c="r")
        plt.plot(dis_q.cpu(), 100.0*q, c="b")
        plt.arrow(mean_tar_q, 50, 0.0, -45, color='r', head_width=0.05, head_length=5, alpha=1.0, width=0.01)
        plt.arrow(mean_dis_q, 50, 0.0, -45, color='b', head_width=0.05, head_length=5, alpha=1.0, width=0.01)
        plt.ylim(0, 100)
        plt.xlim(0, max(tar_q.max().cpu().item(), mean_dis_q.max().cpu().item()))
        plt.savefig(os.path.join(results_folder, f"Percentile_layer_{j}_iter_{i}.svg"), format="svg")
        plt.close()
        # plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
q = torch.linspace(0, 1.0, 51)
tar_m_ = [[] for _ in range(n_layers)]
dis_m_ = [[] for _ in range(n_layers)]
for j in range(n_layers):
    for i in range(n_fix, n_fix_att):
        if i == n_fix:
            tar_m_[j] = targets_[i][j][receptive_[j] > 0.0].ravel()
            dis_m_[j] = distractors_[i][j][receptive_[j] > 0.0].ravel()
        else:
            tar_m_[j] = torch.cat((tar_m_[j], targets_[i][j][receptive_[j] > 0.0].ravel()), dim=0)
            dis_m_[j] = torch.cat((dis_m_[j], distractors_[i][j][receptive_[j] > 0.0].ravel()), dim=0)


for j in range(n_layers):
    tar_q = torch.quantile(tar_m_[j], q.to(DeVice))
    dis_q = torch.quantile(dis_m_[j], q.to(DeVice))
    mean_tar_q = torch.quantile(tar_m_[j], 0.5).cpu()
    mean_dis_q = torch.quantile(dis_m_[j], 0.5).cpu()
    plt.figure(figsize=(6, 4))
    plt.title(f"Layer {j}, Iteration {n_fix}-{n_fix_att}, {mean_tar_q:.2f}, {mean_dis_q:.2f}")
    plt.plot(tar_q.cpu(), 100.0*q, c="r")
    plt.plot(dis_q.cpu(), 100.0*q, c="b")
    plt.arrow(mean_tar_q, 50, 0.0, -45, color='r', head_width=0.05, head_length=5, alpha=1.0, width=0.01)
    plt.arrow(mean_dis_q, 50, 0.0, -45, color='b', head_width=0.05, head_length=5, alpha=1.0, width=0.01)
    plt.ylim(0, 100)
    plt.xlim(-max(tar_q.max().cpu().item(), mean_dis_q.max().cpu().item())/5, max(tar_q.max().cpu().item(), mean_dis_q.max().cpu().item()))
    plt.savefig(os.path.join(results_folder, f"Percentile_layer_{j}.svg"), format="svg")
    plt.close()
    # plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
curve_tar_act = [[] for _ in range(model.n_convs)]
curve_dis_act = [[] for _ in range(model.n_convs)]
for j in range(n_layers):
    for i in range(n_iter):
        curve_tar_act[j].append(targets_[i][j][receptive_[j] > 0.0].mean().clone())
        curve_dis_act[j].append(distractors_[i][j][receptive_[j] > 0.0].mean().clone())
curve_tar_act = torch.tensor(curve_tar_act)
curve_dis_act = torch.tensor(curve_dis_act)
plot_curves(n_layers, curve_tar_act, curve_dis_act, results_folder, "Curve_layer")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
curve_tar_act = [[] for _ in range(model.n_convs)]
curve_dis_act = [[] for _ in range(model.n_convs)]
for j in range(n_layers):
    for i in range(n_iter):
        curve_tar_act[j].append((targets_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
        curve_dis_act[j].append((distractors_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
curve_tar_act = torch.tensor(curve_tar_act)
curve_dis_act = torch.tensor(curve_dis_act)
plot_curves(n_layers, curve_tar_act, curve_dis_act, results_folder, "CurveDeBoth_layer")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
curve_tar_act = [[] for _ in range(model.n_convs)]
curve_dis_act = [[] for _ in range(model.n_convs)]
for j in range(n_layers):
    for i in range(n_iter):
        curve_tar_act[j].append((targets_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
        curve_dis_act[j].append((distractors_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
curve_tar_act = torch.tensor(curve_tar_act)
curve_dis_act = torch.tensor(curve_dis_act)
curve_tar_act = curve_tar_act - curve_tar_act[:, :2].mean(dim=1, keepdim=True)
curve_dis_act = curve_dis_act - curve_dis_act[:, :2].mean(dim=1, keepdim=True)
plot_curves(n_layers, curve_tar_act, curve_dis_act, results_folder, "CurveDeBothDe_layer")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
curve_tar_act = [[] for _ in range(model.n_convs)]
curve_dis_act = [[] for _ in range(model.n_convs)]
for j in range(n_layers):
    for i in range(n_iter):
        curve_tar_act[j].append((targets_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
        curve_dis_act[j].append((distractors_[i][j] - both_[j])[receptive_[j] > 0.0].mean().clone())
curve_tar_act = torch.tensor(curve_tar_act)
curve_dis_act = torch.tensor(curve_dis_act)
curve_tar_act = curve_tar_act - curve_tar_act[:, :2].mean(dim=1, keepdim=True)
curve_dis_act = curve_dis_act - curve_dis_act[:, :2].mean(dim=1, keepdim=True)
curve_tar_act = curve_tar_act / curve_tar_act.max(dim=1, keepdim=True).values
curve_dis_act = curve_dis_act / curve_tar_act.max(dim=1, keepdim=True).values
plot_curves(n_layers, curve_tar_act, curve_dis_act, results_folder, "CurveDeBothDeNorm_layer")
