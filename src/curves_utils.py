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
from src.composer import CurveTracing, transforms, conv2d, bezier_generator
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
    x[:, p-2:p+1, pp:3*pp+p] = 1.0
    c[:, p-3:p+2, pp-2:pp+3] = 1.0
    x = transforms.functional.rotate(x, theta, interpolation=transforms.InterpolationMode.BILINEAR)
    c = transforms.functional.rotate(c, theta, interpolation=transforms.InterpolationMode.BILINEAR)
    x = blur(x)
    c = blur(c)
    x = scale_(x, dim=(-2, -1))
    c = scale_(c, dim=(-2, -1))
    return x, c

def generate_bezier_rot(b: int, h: float, w: float, ds: CurveTracing):
    radius = 0.3
    bezier_order = 1
    resolution = ds.resolution
    stimulus_kernel = ds.stimulus_kernel
    fixate_kernel = ds.fixate_kernel
    saccade_kernel = ds.saccade_kernel
    samples = torch.zeros(3, b, 1, h, w)
    t_ = torch.linspace(0., 1.0, resolution)  # parameter
    c_ = torch.rand(b, bezier_order + 1, 2, 1)  # coordinates
    for j in range(b):
        c_[j, 0, 0] = 0.5 + radius * math.cos(math.pi * j / b)
        c_[j, 0, 1] = 0.5 + radius * math.sin(math.pi * j / b)
        c_[j, 1, 0] = 0.5 - radius * math.cos(math.pi * j / b)
        c_[j, 1, 1] = 0.5 - radius * math.sin(math.pi * j / b)
    for i in range(b):
        im_ = torch.zeros(3, 1, h, w)
        b_ = bezier_generator(c_[i], t_)  # Bezier curve
        b_h, b_w = (b_[0] * h).int().tolist(), (b_[1] * w).int().tolist()
        im_[0, :, b_h, b_w] = 1.0
        im_[1, :, b_h[0], b_w[0]] = 1.0
        im_[2, :, b_h[-1], b_w[-1]] = 1.0
        im_[0] = (conv2d(im_[0], stimulus_kernel, padding='same') > 0.0)
        im_[1] = (conv2d(im_[1], fixate_kernel, padding='same') > 0.0)
        im_[2] = (conv2d(im_[2], saccade_kernel, padding='same') > 0.0)
        samples[:, i] = im_
    return samples

def make_stimuli(h: int, w: int, b: int, ds:CurveTracing , bar: bool = False):
    assert h%4 == 0 and w%4 == 0
    x  = torch.zeros(b, 1, h, w)
    xc = torch.zeros(b, 1, h, w)
    zc = torch.zeros(b, 1, h, w)
    t = torch.zeros(b, 1, h, w)
    theta = torch.linspace(0., 179., b)
    better_bars = generate_bezier_rot(b, h//2, w//2, ds)
    for i in range(b):
        if bar:
            xx, cc = better_bars[0, i], better_bars[1, i] # make_bar(h//2, w//2, theta[i].item())
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
    plt.savefig(os.path.join(results_folder, 'Tuning_Curve' + plotname + '.svg'))
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


class FitBellCurve:
    def __init__(self):
        self.d = 180
        self.x = torch.linspace(0, 179, self.d)
        self.z = torch.ones_like(self.x) * 0.5

    def roll(self, y: torch.Tensor, b: int):
        h = self.d//2
        return y.roll(h - b)

    def normal(self, mu: float, std: float):
        # Normalized Gaussian function
        return torch.exp(-((self.x-mu)**2)/(2*std**2)) / (std * math.sqrt(2*math.pi))

    def mean_euc(self, y_t: torch.Tensor, y_p: torch.Tensor):
        # Calculate the mean euclidean distance between two curves
        return (y_t - y_p).square().sum().sqrt() / self.d

    def __call__(self, y: torch.Tensor):
        d = y.min().item()  # Asymptote (baseline)
        y = y - d  # Remove the asymptote
        a = y.max().item()  # Amplitude (peak)
        y = y / a   # Normalize the amplitude
        b = y.argmax().item()  # Preferred orientation (peak position)
        y = self.roll(y, b)  # center the curve on the preferred orientation
        idx = torch.argwhere((y - self.z).sign().diff())
        # Find the half width at half maximum to calculate the standard deviation
        if idx.numel() > 1:
            c = (idx[1].item() - idx[0].item()) / (2 * math.sqrt(2 * math.log(2)))
        else:
            c = self.d//2
        yy = self.normal(self.d//2, c)  # Generate a normal curve for the given width
        yy = yy / yy.max()  # give the fitted curve the same amplitude as the original
        yy = yy + d  # Add same baseline to the fitted curve 
        e = self.mean_euc(y, yy)  # Calculate the mean euclidean distance between the curves
        return (a, b, c, d), e


class Roelfsema:
    def __init__(self, model, tasks, logger):
        import matplotlib.pyplot as plt
        self.task_name = 'CurveTracing' if 'CurveTracing' in tasks else 'SwitchBox'
        self.model = model
        self.tasks = tasks
        self._loader = tasks[self.task_name]["dataloaders"][-1]
        self._loader.dataset.training = False
        self.n_samples = len(self._loader.dataset)
        self.logger = logger

        fix_attend_saccade = self.tasks[self.task_name]["params"]["fix_attend_saccade"]
        n_iter = sum(fix_attend_saccade)
        n_layers = model.n_convs
        n_fix, n_att, n_sac = fix_attend_saccade
        n_fix_att = n_fix + n_att

    def test_accuracy_curve(self, device):
        correct = 0
        threshold = 0.25
        self._loader = self.tasks[self.task_name]["dataloaders"][-1]
        self._loader.dataset.training = False
        
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for x, _, m, _, c in self._loader:
                    b = x.size(0)
                    x, m, c = x.to(device), m.to(device), c.to(device)
                    p_m, _, _ = self.model(x, self.tasks[self.task_name]["key"])
                    m_correct = (p_m[:, -1] * c[:, 2])
                    m_false = (p_m[:, -1] * c[:, 5])
                    m_correct = m_correct.view(b, -1).sum(dim=-1) / c[:, 2].view(b, -1).sum(dim=-1)
                    m_false = m_false.view(b, -1).sum(dim=-1) / c[:, 5].view(b, -1).sum(dim=-1)
                    correct += ((m_correct - m_false) > threshold).sum().item()
        self.logger.info(f"Test Final Accuracy: {correct / self.n_samples}")
        self._loader.dataset.training = True
