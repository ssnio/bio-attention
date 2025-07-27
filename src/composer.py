# # built-in modules
import random
import os
from typing import Callable, Union
import math
import csv
import pickle
# # Torch modules
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.nn.functional import one_hot, conv2d
# # other modules
from PIL import Image as PILImage


def make_frame(h: int, w: int, t: int) -> torch.Tensor:
    """h: height, w: width, t: thickness
    """
    x = torch.zeros(1, h, w)
    x[:, :t, :] = 1.0
    x[:, -t:, :] = 1.0
    x[:, :, :t] = 1.0
    x[:, :, -t:] = 1.0
    return x

def natural_noise(h, w):
    return torch.cat((
            torch.normal(0.485, 0.229, (1, h, w)),
            torch.normal(0.456, 0.224, (1, h, w)),
            torch.normal(0.406, 0.225, (1, h, w))),
        dim=0)

def gaussian_patch(h: int, w: int, s: float) -> torch.Tensor:
    x = torch.arange(-w // 2 + 1, w // 2 + 1)
    y = torch.arange(-h // 2 + 1, h // 2 + 1)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / s ** 2)
    return zz

def gaussian_line(w: int, m: int, s: int) -> torch.Tensor:
    x = torch.linspace(0, w, w)
    y = torch.exp(-(x - m)**2 / s)
    return x, y

def blur_edges(h: int, w: int, nh: int, nw: int, s: int) -> torch.Tensor:
    xh = torch.ones(h)
    dh = h//nh
    for i in range(0, nh+1):
        _, z = gaussian_line(h, i * dh, s)
        xh *= (1.0 - z)
    hh = xh.unsqueeze(0).repeat(w, 1).t()
    xw = torch.ones(w)
    dw = w//nw
    for i in range(0, nw+1):
        _, z = gaussian_line(w, i * dw, s)
        xw *= (1.0 - z)
    ww = xw.unsqueeze(0).repeat(h, 1)
    return hh * ww

def do_n_it(x: Union[torch.Tensor, int], n: int):
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0).expand(n, -1, -1, -1)
    elif isinstance(x, int):
        return torch.tensor([x]).long().expand(n)
    elif isinstance(x, (list, tuple)):
        return torch.tensor([list(x)]).long().expand(n, -1)
    else:
        raise ValueError(f"Invalid type {type(x)}")


def get_classes(dataset_: Dataset, n: int = 10):
    # get the indices of each class
    classes = [[] for _ in range(n)]
    for i, (_, y) in enumerate(dataset_):
        classes[y].append(i)
    return classes


def pink(shape: tuple, c: float, a: float, actfun: Callable = None):
    # Creates a 2D correlated noise with the given shape
    # c: exponent, a: amplitude, actfun: activation function over the image
    if len(shape) == 2:
        pink = torch.zeros((1, *shape))
    elif len(shape) == 3:
        pink = torch.zeros(shape)
    else:
        return
    if actfun is None:
        actfun = lambda y: y
    if actfun == "scale":
        actfun = lambda y: (y - y.min())/(y.max() - y.min() + 1e-6)
    channels, height, width = pink.shape
    f_height = torch.linspace(1/height, 1.0, height)
    f_width = torch.linspace(1/width, 1.0, width)
    freq = torch.stack(torch.meshgrid((f_height, f_width), indexing="ij")).sum(dim=0) / 2
    for i in range(channels):
        phases = 2 * torch.pi * (torch.rand((height, width)) - 0.5)
        combi = a * (phases * (1 + 1j)) / freq.pow(c)
        pink[i] = torch.fft.ifft2(combi).real
    pink = pink.squeeze()
    return actfun(pink)


def bezier_generator(p: torch.Tensor, t: torch.Tensor):
    """General Bézier curves
    p: points
    t: line range (something like: torch.linspace(0.0, 1.0, 100))
    """
    assert torch.all((0.0 <= t) * (t <= 1.0)), "`t` must be between 0.0 and 1.0 !"
    assert p.shape[1] == 2, "2D curve"
    assert p.shape[2] == 1, "singleton dimension"
    n = p.shape[0] - 1  # order of Bezier curve
    c = ((1.0 - t) ** n) * p[0] # first term
    for i in range(1, n + 1):
        c += math.comb(n, i) * ((1.0 - t) ** (n - i)) * (t ** i) * p[i]
    return c


def center_of_mass(x: torch.tensor):
    """
    :param x: 2-3D tensor
    :return: center of mass of the tensor
    """
    assert x.ndim == 2 or (x.ndim == 3 and x.size(0) == 1)
    d = 10  # min distance from the edges (boundaries)
    h, w = x.shape[-2:]
    eps = 1e-9
    x = x.squeeze()
    normalizer = torch.sum(x) + eps
    grids = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    c = [int(torch.sum(x * grids[dir].float()).item() / normalizer) for dir in range(2)]
    return [max(min(c[0], h - d), d), max(min(c[1], w - d), d)]


def coord_to_points(x: torch.tensor, i: int, j: int, k: int):
    *_, h, w = x.shape
    p = torch.zeros_like(x)
    p[:, max(i-k, 0):min(h, i+k), max(j-k, 0):min(w, j+k)] = 1.0
    return p


def routine_01(composites: torch.Tensor, masks: torch.Tensor, noise: float = 0.0):
    # adding noise and clamping 
    composites += torch.rand(()) * noise * torch.rand_like(composites)
    composites = torch.clamp(composites, 0.0, 1.0)
    masks = torch.clamp(masks, 0.0, 1.0)
    masks = 2.0 * (masks - 0.5)
    return composites, masks


class FixPoints:
    """Creates a fix point for a given object (digit) with size k x k
    such that the fix point is contiguous with the object."""
    def __init__(self, k: int = 3, rand: bool = False):
        self.k = k
        self.rand = rand
        self.p = (k // 2) + (k % 1)
        self.a = k * k # area of the fix points
        self.ck = torch.ones((1, 1, k, k))

    def find_fix_points(self, x: torch.tensor, s: int = 0):
        conv_mask = torch.conv2d(x, self.ck, padding='same')
        return conv_mask.where(conv_mask >= self.a - 1 - s, torch.tensor(0.0)).nonzero()
    
    def get_rand_fix_point(self, x: torch.tensor):
        s, max_s = 0, self.a
        while s < max_s:
            fix_points = self.find_fix_points(x, s)
            if fix_points.size(0) > 0:
                break
            s += 1
        _, i, j = random.choice(fix_points) if s < max_s else [None, torch.zeros(self.p), torch.zeros(self.p)]
        return [i.item(), j.item()]
    
    def fix_points(self, x: torch.tensor):
        fix_points = torch.zeros_like(x)
        if self.rand:
            i, j = torch.randint(self.k, x.size(1) - self.k, (1,)), torch.randint(self.k, x.size(2) - self.k, (1,))
        else:
            i, j = self.get_rand_fix_point(x)
        fix_points[:, i-self.p:i+self.p, j-self.p:j+self.p] = 1.0
        return fix_points


class StringDigits(Dataset):
    def __init__(self, mnist_ds: Dataset):
        self.ds = mnist_ds
        self.h, self.w = 14, 140
        self.transform = transforms.Resize((self.h, self.h), antialias=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        composite = torch.zeros(3, self.h, self.w)
        for i in range(self.w//self.h):
            idx = torch.randint(0, len(self.ds), (1,)).item()
            x, _ = self.ds[idx]
            x = self.transform(x)
            composite[:, :, i*self.h:(i+1)*self.h] = x 
        return composite


class IOR_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,
                 n_digits: int,  # number of digits
                 n_attend: int,  # number of attend iterations per digit
                 noise: float = 0.25,  # noise scale
                 overlap: float = 0.0,  # maximum permissible overlap between digits
                 ):
        super().__init__()
        self.dataset = mnist_dataset
        self.n_digits = n_digits
        self.n_attend = n_attend
        self.n_iter = n_digits * n_attend
        self.noise = noise
        self.overlap = overlap
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(0.9, 1.1))])

    def build_valid_test(self):
        """The noise is set to zero for validation and testing."""
        self.noise = 0.0

    def get_random_digit(self):
        i = torch.randint(0, self.dataset.__len__(), (1,)).item()
        return self.dataset.__getitem__(i)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(self.n_digits, self.c, self.h, self.w)
        masks = torch.zeros(self.n_digits, 1, self.h, self.w)
        labels = torch.zeros(self.n_digits).long()
        hot_labels = 0
        composites += (0.5 * torch.rand(1, 3, 1, 1))
        
        # composing while avoiding overlap
        for i in range(self.n_digits):
            j, max_j = 0, 17
            while j < max_j:
                x, y = self.get_random_digit()
                x = self.transform(x)
                if (components.sum(0) * x).sum() <= self.overlap:
                    break
                j += 1
            rgb = torch.rand(3, 1, 1)  # random.choice(self.rgbs)
            rgb /= rgb.max()
            composites[:] = (1.0 - x) * composites + x * rgb
            components[i] = x
            labels[i] = y
            masks[i] = x
        
        # adding noise and clamping
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Arrow_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 directory: str = r"./data/",  # directory of the arrow images
                 exclude: bool = True,  # whether to exclude the digit class from the random samples
                 ):
        
        super().__init__()
        self.directory = os.path.join(directory, "arrows")
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.noise = noise
        self.exclude = exclude
        self.pad = 2
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = 96, 96
        self.transform = transforms.Pad(self.pad)
        self.n_classes = 10
        self.class_ids = get_classes(self.dataset, 10)
        self.arrows = self.get_arrows()
        self.arrow_pos = [5, 0, 4,
                          2,    1,
                          7, 3, 6]
        self.digit_pos = [(0,  0), (0,  32), (0,  64),
                          (32, 0),           (32, 64),
                          (64, 0), (64, 32), (64, 64)]

    def rand_sample(self, y: int, exclude: bool = False):
        if exclude:
            t = list(range(10))
            t.remove(y)
            cls = random.choice(t)
        else:
            # cls = y
            cls = random.choice(list(range(10)))
        i = self.class_ids[cls][torch.randint(0, len(self.class_ids[cls]), (1, )).item()]
        return self.dataset.__getitem__(i)

    def get_arrows(self):
        arrows = torch.zeros(8, 1, 32, 32)
        for i, file in enumerate(["n", "e", "w", "s", "ne", "nw", "se", "sw"]):
            arrows[i, :, 1:-1, 1:-1] = 1.0 * (transforms.ToTensor()(PILImage.open(os.path.join(self.directory, f"{file}.png")))[0] > 0.0)
        return arrows

    def build_valid_test(self):
        self.exclude = False
        self.noise = 0.0

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        rgbs = torch.rand(9, 3, 1, 1)
        labels[:] = y
        hot_labels = 0
        b_rgb = torch.rand(3, 1, 1) * 0.5

        t = random.choice(range(8))
        arrow_i, pos_ij = self.arrow_pos[t], self.digit_pos[t]
        composites[:, :, 32:64, 32:64] = self.arrows[arrow_i] * rgbs[0] + (1.0 - self.arrows[arrow_i]) * b_rgb
        composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[1] + (1.0 - x) * b_rgb
        masks[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] += x
        j = 2
        for i in range(8):
            if i != t:
                x, _ = self.rand_sample(y, exclude=self.exclude)
                x = self.transform(x)
                pos_ij = self.digit_pos[i]
                composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[j] + (1.0 - x) * b_rgb
                j += 1
        
        # adding noise and clamping
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Cue_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 fix_attend: tuple,  # number of fixate and attend iterations
                 n_digits: int,  # number of digits
                 noise: float = 0.25,  # noise scale
                 overlap: float = 0.0,  # maximum permissible overlap between digits
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.fixate, self.attend = fix_attend
        self.n_iter = sum(fix_attend)
        self.n_digits = n_digits
        self.noise = noise
        self.overlap = overlap
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(0.9, 1.1))])
        self.fix_pointer = FixPoints(5)

    def build_valid_test(self):
        self.noise = 0.0

    def get_random_digit(self):
        i = torch.randint(0, self.dataset.__len__(), (1,)).item()
        return self.dataset.__getitem__(i)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(self.n_digits, self.c, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = 0
        y_list = []
        composites += (0.5 * torch.rand(1, 3, 1, 1))

        # composing while avoiding overlap
        for i in range(self.n_digits):
            j, max_j = 0, 17
            while j < max_j:
                x, y = self.get_random_digit()
                x = self.transform(x)
                if (components.sum(0) * x).sum() <= self.overlap:
                    break
                j += 1
            # rgb = random.choice(self.rgbs)
            components[i] += x
            y_list.append(y)
        target_id = random.choice(range(self.n_digits))
        target = components[target_id]
        fixpoint = self.fix_pointer.fix_points(target)
        composites[:self.fixate] = fixpoint + (1.0 - fixpoint) * composites[:self.fixate]
        masks[:self.fixate] = fixpoint
        rgb = torch.rand(self.n_digits, 3, 1, 1)
        rgb /= rgb.max(dim=1, keepdims=True).values
        composites[self.fixate:] = (components * rgb).sum(0) + (1.0 - components.sum(0)) * composites[self.fixate:]
        masks[self.fixate:] = target
        labels[:] = y_list[target_id]
        
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Recognition_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 stride: int,  # stride of background and foreground motion per iteration
                 blank: bool = False,  # whether to the foreground is visible (blank) or not
                 static: bool = False,  # whether the background and foreground are static
                 background: bool = True,  # whether to use background or not
                 noise: float = 0.25,  # noise scale
                 hard: bool = False,
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.stride = stride
        self.blank = blank
        self.static = static
        self.background = background
        self.noise = noise
        self.hard = hard
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(1.3, 1.5))])
        self.occ_ratio = 0.0
        self.n = 0

    def make_foreground(self):
        window_shape = (self.h, self.w + self.stride * self.n_iter)
        pink_fore = pink(window_shape, 2.5, 0.6, torch.cos).unsqueeze(0)
        pink_fore = 1.0 * ((0.5 + pink_fore / 2) > 0.7)
        return pink_fore

    def make_background(self):
        window_shape = (self.h, self.w + self.stride * self.n_iter)
        a, b, c = torch.rand(3)
        pink_back = pink(window_shape, 2.0 + a, 0.25 + 0.5 * b, torch.cos).unsqueeze(0)
        pink_back = 1.0 * ((0.5 + pink_back / 2) > 0.25 + 0.5 * c)
        return pink_back

    def build_valid_test(self):
        self.noise = 0.0

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.transform(x)
        
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = 0

        # get background and foreground
        digit_color = torch.rand(3, 1, 1)
        background_color = torch.rand(3, 1, 1)
        foreground_color = 1.0 - background_color
        if not self.hard:
            foreground_color /= (foreground_color.max() + 1e-6) * 2.0
            background_color /= (background_color.max() + 1e-6) * 2.0
        digit_color /= (digit_color.max() + 1e-6)
        if self.background:
            background = self.make_background() * background_color
        else:
            background = torch.zeros(3, self.h, self.w + self.stride * self.n_iter)
        foreground = self.make_foreground()

        # assignment
        labels[:] = y
        masks[:] = x
        z = digit_color * x
        self.occ_ratio += (1.0 - (x * (1.0 - foreground[:, :self.h, :self.w])).sum() / x.sum())
        self.n += 1

        # background
        if self.static:
            composites[:] = background_color * background[:, :, :self.w] * (1.0 - x) + z
        else:
            r_stride = random.randint(1, self.stride)
            d = random.randint(0, 1)
            for i in range(self.n_iter):
                if d == 0:
                    slc = slice(i * r_stride, self.w + i * r_stride)
                else:
                    slc = slice((self.n_iter - i) * r_stride, self.w + (self.n_iter - i) * r_stride)
                composites[i] = background_color * background[:, :, slc] * (1.0 - x) + z

        # foreground
        r_stride = random.randint(1, self.stride)
        d = random.randint(0, 1)
        for i in range(self.n_iter):
            if self.static:
                slc = slice(0, self.w)
            else:
                if d == 0:
                    slc = slice(i * r_stride, self.w + i * r_stride)
                else:
                    slc = slice((self.n_iter - i) * r_stride, self.w + (self.n_iter - i) * r_stride)
            if self.blank:
                composites[i] = (composites[i] * (1.0 - foreground[:, :, slc]))
            else:
                composites[i] = (composites[i] * (1.0 - foreground[:, :, slc])) + foreground_color * foreground[:, :, slc]

        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Search_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 n_digits: int,  # number of digits
                 noise: float = 0.25,  # noise scale
                 overlap: float = 1.0,  # maximum permissible overlap between digits
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.n_digits = n_digits
        self.noise = noise
        self.overlap = overlap
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(0.9, 1.1))])
        self.classes = get_classes(self.dataset, 10)

    def build_valid_test(self):
        self.noise = 0.0

    def get_random_digit(self, ind_list: list = None):
        if ind_list is None:
            i = torch.randint(0, self.dataset.__len__(), (1,)).item()
        else:
            i = random.choice(ind_list)
        return self.dataset.__getitem__(i)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(self.n_digits, self.c, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = torch.zeros(self.n_iter, 10).float()
        y_list = []
        b_rgb = torch.rand(3, 1, 1) * 0.5

        # composing while avoiding overlap
        rand_classes = random.sample(self.classes, k=self.n_digits)
        for i, s in enumerate(rand_classes):
            j, max_j = 0, 17
            while j < max_j:
                x, y = self.get_random_digit(s)
                x = self.transform(x)
                if (components.sum(0) * x).sum() <= self.overlap:
                    break
                j += 1
            components[i] += x
            y_list.append(y)
        
        rgb = torch.rand(self.n_digits, 3, 1, 1)
        rgb /= rgb.max(dim=1, keepdims=True).values
        composites[:] = (components * rgb).sum(0) + (1.0 - components.sum(0)) * b_rgb
        
        # selecting the target
        target_id = random.choice(range(self.n_digits))
        target_label = y_list[target_id]
        masks[:] = components[target_id]

        labels[:] = target_label
        hot_labels[:] = one_hot(labels[0], 10).squeeze().float()
 
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Tracking_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 fix_attend: tuple = (2, 5),  # number of fixate and attend iterations
                 n_digits: int = 3,  # number of distractor images
                 noise: float = 0.25,  # noise scale
                 ):
        
        super().__init__()
        assert n_digits < 8
        self.dataset = mnist_dataset  # MNIST datasets
        self.fix_attend = fix_attend
        self.fix, self.attend = fix_attend  # [fixation, attention]
        self.n_iter = sum(fix_attend)  # number of episodes (recurrent iterations)
        self.n_digits = n_digits  # number of distractor images 
        self.noise = noise  # noise scale
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.classes = get_classes(self.dataset, 10)
        self.zones = {"nw": ((0,  22), (0,  22), (            "ne",       "ee", "sw", "ss", "se")),
                      "nn": ((0,  22), (23, 55), (                              "sw", "ss", "se")),
                      "ne": ((0,  22), (56, 68), ("nw",             "ww",       "sw", "ss", "se")),
                      "ww": ((23, 55), (0,  22), (            "ne",       "ee",             "se")),
                      "ee": ((23, 55), (56, 68), ("nw",             "ww",       "sw",           )),
                      "sw": ((56, 68), (0,  22), ("nw", "nn", "ne",       "ee",             "se")),
                      "ss": ((56, 68), (23, 55), ("nw", "nn", "ne",                             )),
                      "se": ((56, 68), (56, 68), ("nw", "nn", "ne", "ww",       "sw",           )),
                      }
        self.bezier_order, self.bezier_res = 2, min(self.h, self.w)
        self.bezier_inds = torch.linspace(0, self.bezier_res - 1, self.attend).long()
        
    def pick_start_end(self, sz: str):
        """sz: start zone
        """
        si = random.randint(*self.zones[sz][0])  # start i
        sj = random.randint(*self.zones[sz][1])  # start j
        ez = random.choice(self.zones[sz][2])  # end zone
        ei = random.randint(*self.zones[ez][0])  # end i
        ej = random.randint(*self.zones[ez][1])  # end j
        return (si, sj), (ei, ej)

    def make_path(self, start, end):
        t_ = torch.linspace(0., 1.0, self.bezier_res)  # parameter
        c_ = torch.rand(self.bezier_order + 1, 2, 1)  # coordinates
        c_[0] = torch.tensor([start[0]/self.h, start[1]/self.w]).view(2, 1)
        c_[-1] = torch.tensor([end[0]/self.h, end[1]/self.w]).view(2, 1)
        b_ = bezier_generator(c_, t_)  # Bezier curve
        b_h, b_w = (b_[0] * self.h).long().tolist(), (b_[1] * self.w).long().tolist()
        return b_h, b_w
    
    def put_digits_on_curve(self, z, x, c, b_h, b_w):
        """
        z: composite (self.n_iter, 3, self.h, self.w)
        x: digit (1, 28, 28)
        c: color (3, 1, ,1)
        b_h: h (i) indices (self.bezier_res)
        b_w: w (j) indices  (self.bezier_res)
        """
        for k, b in enumerate(self.bezier_inds):
            ih, iw = min(b_h[b], self.h-28), min(b_w[b], self.w-28)
            z[k+self.fix, :, ih:ih+28, iw:iw+28] += x * c
        return z

    def build_valid_test(self):
        self.noise = 0.0

    def get_random_digit(self, ind_list: list = None):
        if ind_list is None:
            i = torch.randint(0, self.dataset.__len__(), (1,)).item()
        else:
            i = random.choice(ind_list)
        return self.dataset.__getitem__(i)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)

        # coloring
        rgbs = torch.rand(self.n_digits, 3, 1, 1)
        rgbs /= torch.linalg.norm(rgbs, dim=1, keepdim=True)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = 0
        labels[:] = y

        # location
        positions = []
        s_zones = random.sample(sorted(self.zones), self.n_digits)
        for sz in s_zones:
            (si, sj), (ei, ej) = self.pick_start_end(sz)
            b_h, b_w = self.make_path((si, sj), (ei, ej))
            positions.append((b_h, b_w))

        # assignment
        i, j = positions[0][0][0], positions[0][1][0]
        composites[:self.fix, :, i:i+28, j:j+28] = x * rgbs[0]
        composites = self.put_digits_on_curve(composites, x, rgbs[0], *positions[0])
        masks = 1.0 * (composites.sum(dim=1, keepdim=True) > 0.0)
        for i in range(1, self.n_digits):
            x, y = self.get_random_digit()
            composites = self.put_digits_on_curve(composites, x, rgbs[i], *positions[i])

        where_digits = composites * (composites > 0.0)
        composites = 0.5 * (1.0 - where_digits) * torch.rand(1, 3, 1, 1) + where_digits
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Popout_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 exclude: bool = True,  # whether to exclude the digit class from the random samples
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.noise = noise
        self.exclude = exclude
        self.pad = 2
        self.h, self.w = 96, 96
        self.transform = transforms.Pad(self.pad)
        self.n_classes = 10
        self.class_ids = get_classes(self.dataset, 10)
        self.index_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.digit_pos = [(0,  0), (0,  32), (0,  64),
                          (32, 0), (32, 32), (32, 64),
                          (64, 0), (64, 32), (64, 64)]

    def rand_sample(self, y: int, exclude: bool = False):
        if exclude:
            t = list(range(10))
            t.remove(y)
            cls = random.choice(t)
        else:
            # cls = y
            cls = random.choice(list(range(10)))
        i = self.class_ids[cls][torch.randint(0, len(self.class_ids[cls]), (1, )).item()]
        return self.dataset.__getitem__(i)

    def build_valid_test(self):
        self.exclude = False
        self.noise = 0.0

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        rgbs = torch.rand(2, 3, 1, 1)
        labels[:] = y
        hot_labels = 0
        b_rgb = 0.5 * torch.rand(3, 1, 1)

        t = random.choice(self.index_pos)
        pos_ij = self.digit_pos[t]
        composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = (x * rgbs[0]) + (1.0 - x) * b_rgb
        masks[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] += x
        
        x, _ = self.rand_sample(y, exclude=self.exclude)
        x = self.transform(x)
        for i in self.index_pos:
            if i != t:
                pos_ij = self.digit_pos[i]
                composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[1] + (1.0 - x) * b_rgb
        
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class PatternedRecognition_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.noise = noise
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.patterns = Patterns(self.h, self.w)
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(1.3, 1.5))])
        self.occ_ratio = 0.0
        self.n = 0

    def build_valid_test(self):
        self.noise = 0.0

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.transform(x)
        
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = 0

        # assignment
        labels[:] = y
        masks[:] = x

        # get background and foreground
        digit_color = torch.rand(3, 1, 1)
        background_color = torch.rand(3, 1, 1)
        background_color /= (background_color.max() + 1e-6) * 2
        digit_color /= (digit_color.max() + 1e-6)
        background = self.patterns.__getitem__(0)
        composites[:] = (x * background) * digit_color + ((1.0 - x) * background) * background_color

        self.occ_ratio += ((x * (1.0 - background)).sum() / x.sum())
        self.n += 1

        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class LineRecognition_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.h, self.w = 96, 96
        self.hh, self.ww = 42, 42
        self.hor = torch.tensor([[1, 1, 1, 1],
                                 [0, 0, 0, 0],
                                 [1, 1, 1, 1],
                                 [0, 0, 0, 0]]).float().reshape(1, 4, 4).repeat(1, 96//4, 96//4)
        self.ver = self.hor.permute(0, 2, 1)
        self.patterns = torch.stack([self.hor, self.ver])
        self.resize = transforms.Resize((self.hh, self.ww), antialias=True)
    def build_valid_test(self):
        pass

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.resize(x)
        x = (x > 0.5).float()
        t = torch.zeros(1, self.h, self.w)
        p_top = torch.randint(0, self.h - self.hh, (1,)).item()
        p_left = torch.randint(0, self.w - self.ww, (1,)).item()
        t[:, p_top:p_top+self.hh, p_left:p_left+ self.ww] = x
        
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = 0

        # assignment
        labels[:] = y
        masks[:] = t

        # get background and foreground
        color = torch.rand(3, 1, 1)
        rand_txt = torch.randperm(2)
        (background, foreground) = self.patterns[rand_txt]
        composites[:] = color * ((t * foreground) + (1 - t) * background)

        # adding noise and clamping 
        composites = torch.clamp(composites, 0.0, 1.0)
        masks = torch.clamp(masks, 0.0, 1.0)
        masks = 2.0 * (masks - 0.5)

        return composites, labels, masks, components, hot_labels


class CelebACrop(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 n_iter: int,
                 hair_dir: str = None,
                 in_dims: tuple = (3, 178, 178),
                 padding: int = 19,
                 noise: float = 0.125,
                 kind: str = "train",
                 which: int = 0,):
        super().__init__()
        self.dataset = dataset
        self.n_iter = n_iter
        self.hair_dir = hair_dir if hair_dir is not None else r"./data"
        self.in_dims = in_dims
        _, self.h, self.w = self.in_dims
        self.padding = padding
        self.noise = noise if kind == "train" else 0.0
        self.kind = kind  # train, valid, test
        self.which = which  # 0: all, 1: fblonde, 2: fblack, 3: mblonde, 4: mblack
        self.which_names = ["all", "fblonde", "fblack", "mblonde", "mblack"]
        self.len_which = len(self.which_names)
        assert self.which in range(self.len_which), f"which must be between 0 and {self.len_which-1} but got {self.which}!"
        self.at_list = ['Male', 'Black_Hair', 'Blond_Hair']
        self.gender_i, self.black_i, self.blonde_i =(self.dataset.attr_names.index(x) for x in self.at_list)
        self.hair_ids = self.get_hair()
        self.transform = transforms.Compose([
            transforms.Resize((self.h - 2*self.padding, self.w - 2*self.padding), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5 if kind == "train" else 0.0),
            # transforms.RandomVerticalFlip(p=0.5 if kind == "train" else 0.0),
            ])

    def get_hair(self):
        if self.hair_dir and os.path.exists(os.path.join(self.hair_dir, f"celeba/{self.kind}_hair_ids.pt")):
            print(f'Loading {self.kind}_hair_ids.pt from file!')
            return torch.load(os.path.join(self.hair_dir, f"celeba/{self.kind}_hair_ids.pt"))
        else:
            print(f'Creating {self.kind}_hair_ids.pt file!')
            hair_ids = [[], [], [], [], []]  # [all, fblonde, fblack, mblonde, mblack]
            for i, (_, y) in enumerate(self.dataset):
                if y[self.blonde_i] == 1 or y[self.black_i] == 1:
                    hair_ids[0].append(i)
                    if self.kind == "train" and y[self.gender_i] == 1:  # since the dataset is not balanced
                        hair_ids[0].append(i)
                    g, b = y[self.gender_i], y[self.black_i]
                    hair_ids[1+b+2*g].append(i)
        if self.kind == "train":
            random.shuffle(hair_ids[0])  # shuffle the training set only # # # # 
        for i, n in enumerate(self.which_names):
            print(f"{n}: {len(hair_ids[i])}")
        torch.save(hair_ids, os.path.join(self.hair_dir, f"celeba/{self.kind}_hair_ids.pt"))
        print(f'{self.kind}_hair_ids.pt saved to file!')
        return hair_ids

    def __len__(self):
        return len(self.hair_ids[self.which])

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.hair_ids[self.which][idx]]
        x = x[:, 20:-20, :]
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        composites[:] = self.transform(x)
        labels = torch.zeros(self.n_iter).long()
        labels[:] = y[self.gender_i]
        composites += torch.rand(()) * self.noise * torch.rand_like(composites)
        composites = torch.clamp(composites, 0.0, 1.0)
        return composites, labels, 0, 0, 0


class COCOTokens:
    def __init__(self,
                 directory: str,
                 animals: bool = True,
                 split: float = 0.9,
                 ):
        from src.pycocotools.coco import COCO
        self.split = split
        self.directory = os.path.join(directory, "coco")
        self.coco = COCO(os.path.join(self.directory, "annotations/instances_train2017.json"))
        self.coco_test = COCO(os.path.join(self.directory, "annotations/instances_val2017.json"))
        if animals:
            self.entities = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        else:
            self.entities = list(o["name"] for o in self.coco.loadCats(self.coco.getCatIds()))
        self.ids = self.coco.getCatIds(catNms=self.entities)

    def get_tokens(self):
        trvl_tokens = self._get_tokens(self.coco, False)
        test_tokens = self._get_tokens(self.coco_test, True)
        train_tokens, valid_tokens = self._split(trvl_tokens)
        return train_tokens, valid_tokens, test_tokens

    def _split(self, x: Union[list, tuple]):
        n_train = int(len(x) * self.split)
        return (x[:n_train], x[n_train:])

    def _get_tokens(self, ds_, test: bool = False):
        tokens = []
        for id in self.ids:
            tokens += ds_.getImgIds(catIds=[id])
        (258322 in tokens) and tokens.remove(258322)
        (214520 in tokens) and tokens.remove(214520)
        not test and random.shuffle(tokens)
        return torch.tensor(tokens)


class COCOAnimals(Dataset):
    def __init__(self,
                 in_dims: tuple,
                 directory: str,
                 kind: int,  # 0: train, 1: valid, 2: test
                 tokens: torch.Tensor,
                 animals: bool = True,
                 min_area: float = 1.0/64.0,
                 ignore_class_weight: bool = False,
                 ):
        super().__init__()
        in_dims = in_dims if len(in_dims) == 2 else in_dims[1:]
        from src.pycocotools.coco import COCO
        self.h, self.w = in_dims
        self.kind = kind
        self.tokens = tokens
        self.min_area = min_area
        self.ignore_class_weight = ignore_class_weight
        self.directory = os.path.join(directory, "coco")
        self.file_type = "val2017" if kind == 2 else "train2017"
        self.coco = COCO(os.path.join(self.directory, f"annotations/instances_{self.file_type}.json"))
        if animals:
            self.entities = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        else:
            self.entities = list(o["name"] for o in self.coco.loadCats(self.coco.getCatIds()))
        self.ids = self.coco.getCatIds(catNms=self.entities)
        self.n_classes = len(self.entities)
        self.classes = list(range(self.n_classes))
        self.id_to_label = dict(zip(self.ids, self.classes))
        self.transform = transforms.ToTensor()
        self.fix_kernel = torch.ones(1, 1, 7, 7)
        
        self.triplets = None  # labels list of triplets [token, ann_id, label]
        self.tk_cls_anns = None  # list of token, label, list of ann_ids
        self.tk_cls_sann = None  # list of token, label, single ann_id
        self.tk_cls_osann = None  # list of token, label, only single ann_id
        self.class_weights = None
        self._len_ = 0

    def _is_it_good(self, ann: dict, token: int, crowdisok: bool = False):
        height, width = self.coco.loadImgs(token)[0]['height'], self.coco.loadImgs(token)[0]['width']
        category_id = ann['category_id']
        area = ann["area"] / (height * width)
        isnotcrowd = (True, False)[ann['iscrowd']] or crowdisok
        its_good = (category_id in self.ids and isnotcrowd and area > self.min_area)
        return its_good

    def _get_ann(self, ann_id: int):
        return self.coco.loadAnns([ann_id])[0]

    def _load_anns(self, token: int):
        return self.coco.loadAnns(self.coco.getAnnIds([token]))

    def _crop_box(self, x: torch.Tensor, ann: dict):
        x_min, y_min, width, height = ann['bbox']
        x_min, y_min = int(x_min), int(y_min)
        width, height = min(int(width) + 1, x.size(-1)), min(int(height) + 1, x.size(-2))
        return x[:, y_min:y_min+height, x_min:x_min+width]

    def _get_image(self, token: int):
        meta_data = self.coco.loadImgs(token)[0]
        file_name = meta_data["file_name"]
        PIL_file = PILImage.open(os.path.join(self.directory, "images/{}/{}".format(self.file_type, file_name)))
        return self.transform(PIL_file.convert("RGB"))

    def _get_mask(self, ann: dict):
        return (1.0 * (self.transform(self.coco.annToMask(ann)) > 0.0))
        
    def _get_targets(self, token: int, ann_id: int):
        ann_ = self._load_anns(token)[ann_id]
        label = self.id_to_label[ann_["category_id"]]
        mask = self._get_mask(ann_)
        return label, mask

    def _get_class_weights(self):
        assert (self.class_weights > 0.0).all(), "class weights are not set or a class has 0 instance!"
        self.class_weights = len(self.tokens) / self.class_weights
        self.class_weights = self.class_weights / self.class_weights.sum()

    def _get_tokens(self, shuffle: bool = False):
        if self.tokens is None:
            self.tokens = []
            for id in self.ids:
                self.tokens += self.coco.getImgIds(catIds=[id])
            if self.file_type == "train2017":  # # droping the two problematic images
                (258322 in self.tokens) and self.tokens.remove(258322)
                (214520 in self.tokens) and self.tokens.remove(214520)
            shuffle and random.shuffle(self.tokens)
        return self.tokens

    def _get_triplets(self):
        if self.triplets is None:
            self.class_weights = torch.zeros(self.n_classes)
            self.triplets = []
            self.triplet_classes = list([] for _ in range(self.n_classes))
            i = 0
            for token_ in self.tokens:
                token_ = token_.item()
                for ann_ in self._load_anns(token_):
                    if self._is_it_good(ann_, token_):
                        label_ = self.id_to_label[ann_['category_id']]
                        self.triplets.append((token_, ann_['id'], label_))
                        self.triplet_classes[label_].append(i)
                        self.class_weights[label_] += 1
                        i += 1

            self._get_class_weights() if not self.ignore_class_weight else None
        self._len_ = max(len(self.triplets), self._len_)
        return torch.tensor(self.triplets)

    def _get_tk_cls_anns(self, crowdisok: bool = False):
        if self.tk_cls_anns is None:
            self.tk_cls_anns = []
            for token_ in self.tokens:
                token_ = token_.item()
                for cls in self.classes:
                    anns_ids = []
                    for ann_ in self._load_anns(token_):
                        if self._is_it_good(ann_, token_, crowdisok):
                            if self.id_to_label[ann_['category_id']] == cls:
                                anns_ids.append(ann_['id'])
                    if len(anns_ids) > 0:
                        self.tk_cls_anns.append((token_, cls, anns_ids))
        self._len_ = max(len(self.tk_cls_anns), self._len_)
        return self.tk_cls_anns

    def _get_tk_cls_sann(self, crowdisok: bool = False):
        if self.tk_cls_sann is None:
            self.tk_cls_sann = []
            for token_ in self.tokens:
                token_ = token_.item()
                for cls in self.classes:
                    the_ann_id = None
                    a = 0.0
                    for ann_ in self._load_anns(token_):
                        if self._is_it_good(ann_, token_, crowdisok):
                            if self.id_to_label[ann_['category_id']] == cls:
                                if ann_['area'] > a:
                                    the_ann_id = ann_['id']
                                    a = ann_['area']
                    if the_ann_id is not None:
                        self.tk_cls_sann.append((token_, cls, the_ann_id))
        self._len_ = max(len(self.tk_cls_sann), self._len_)
        return torch.tensor(self.tk_cls_sann)

    def _get_tk_cls_osann(self, crowdisok: bool = False):
        if self.tk_cls_osann is None:
            self.tk_cls_osann = []
            for token_ in self.tokens:
                token_ = token_.item()
                for cls in self.classes:
                    anns_ids = []
                    for ann_ in self._load_anns(token_):
                        if self._is_it_good(ann_, token_, crowdisok):
                            if self.id_to_label[ann_['category_id']] == cls:
                                anns_ids.append(ann_['id'])
                    if len(anns_ids) == 1:
                        self.tk_cls_osann.append((token_, cls, anns_ids[0]))
        self._len_ = max(len(self.tk_cls_osann), self._len_)
        return torch.tensor(self.tk_cls_osann)

    def _crop_square(self, x: torch.Tensor):
        _, x_h, x_w = x.shape
        hw = min(x_h, x_w)
        top = torch.randint(0, x_h - hw, (1, )).item() if x_h > hw else 0
        left = torch.randint(0, x_w - hw, (1, )).item() if x_w > hw else 0
        return x[:, top:top+hw, left:left+hw]

    def _resize(self, x: torch.Tensor, h: int, w: int):
        _, x_h, x_w = x.shape
        scale = min(h / x_h, w / x_w)
        if scale < 1.0:
            new_h, new_w = int(x_h * scale) - 1, int(x_w * scale) - 1
            x = transforms.functional.resize(x, size=(new_h, new_w), antialias=False)
        return x

    def _pad(self, x: torch.Tensor, h: int, w: int):
        _, x_h, x_w = x.shape
        scale = max(h - x_h, w - x_w)
        if scale > 0:
            top = 0 if h - x_h == 0 else torch.randint(0, h - x_h, (1, )).item()
            right = 0 if w - x_w == 0 else torch.randint(0, w - x_w, (1, )).item()
            bottom, left = h - x_h - top, w - x_w - right
            padding = tuple([max(0, i) for i in (left, top, right, bottom)])
            x = transforms.functional.pad(x, padding=padding, padding_mode='constant', fill=0.0)
        return x

    def _resize_pad(self, x: torch.Tensor, h: int, w: int, token: int = None, crop: bool = True):
        x = self._crop_square(x) if (crop and h == w and random.random() > 0.9) else x
        x = self._resize(x, h, w)
        x = self._pad(x, h, w)
        if x.shape[-2:] != (h, w):
            print(f"failed for {token}")
            x = torch.randn(x.size(0), h, w)
        return x

    def _getitem_triplet(self, idx):
        token, ann_id, y = self.triplets[idx]
        x = self._get_image(token)
        m = self._get_mask(self._get_ann(ann_id))
        xm = torch.cat((x, m), dim=0)
        xm = self._resize_pad(xm, self.h, self.w, token)
        return xm, y
    
    def _getitem_tk_cls_anns(self, idx):
        token, y, ann_ids = self.tk_cls_anns[idx]
        x = self._get_image(token)
        m = torch.zeros(len(ann_ids), x.size(1), x.size(2))
        for i, ann_id in enumerate(ann_ids):
            m[i] = self._get_mask(self._get_ann(ann_id))
        xm = torch.cat((x, m), dim=0)
        xm = self._resize_pad(xm, self.h, self.w, token)
        return xm, y
    
    def _getitem_tk_cls_sann(self, idx):
        token, y, ann_id = self.tk_cls_sann[idx]
        x = self._get_image(token)
        m = self._get_mask(self._get_ann(ann_id))
        xm = torch.cat((x, m), dim=0)
        xm = self._resize_pad(xm, self.h, self.w, token)
        return xm, y
    
    def _getitem_tk_cls_osann(self, idx):
        token, y, ann_id = self.tk_cls_osann[idx]
        x = self._get_image(token)
        m = self._get_mask(self._get_ann(ann_id))
        xm = torch.cat((x, m), dim=0)
        xm = self._resize_pad(xm, self.h, self.w, token)
        return xm, y


class BG20k(Dataset):
    def __init__(self, root: str, kind: str):
        self.root = os.path.join(root, r"BG-20k")
        self.transform = transforms.ToTensor()
        self.dataset = datasets.ImageFolder(root=self.root, transform=self.transform)
        self.kind = kind
        if kind == 'train':
            self.start, self.end = 0, 15000
        elif kind == 'test' or kind == 'valid':
            self.start, self.end = 15000, 20000

    def build_valid_test(self):
        self.kind = "test"
        self.start, self.end = 15000, 20000

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        x, _ = self.dataset[self.start+idx]
        return x
    

class PerceptualGrouping_COCO(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 fix_attend: tuple,
                 noise: float = 0.25,
                 natural: bool = False):
        
        super().__init__()
        self.kind = "train"
        self.natural = natural
        assert len(fix_attend) == 2
        self.k = 5
        self.dataset = coco_dataset  # COCO datasets
        self.h, self.w = coco_dataset.h, coco_dataset.w
        self.dataset._get_triplets()
        self.class_weights = self.dataset.class_weights
        self.fixate, self.attend = fix_attend
        self.n_iter = sum(fix_attend)
        self.noise = noise  # noise scale
        self.fix_pointer = FixPoints(self.k)
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.125),
            transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomAutocontrast(p=0.125),
            # transforms.RandomInvert(p=0.125),
            ])

    def build_valid_test(self):
        self.kind = "not_train"
        self.flip_transform = lambda x: x
        self.color_transform = lambda x: x
        self.noise = 0.0

    def get_fixed_point(self, m: torch.tensor):
        if torch.rand(()) > 0.5 or self.kind == "not_train":
            ci, cj = center_of_mass(m)
        else:
            ci, cj = self.fix_pointer.get_rand_fix_point(m)
        return coord_to_points(m, ci, cj, self.k)

    def __len__(self):
        return self.dataset._len_ if self.kind == "train" else len(self.dataset.triplets)

    def __getitem__(self, idx: int):
        idx = torch.randint(0, len(self.dataset.triplets), (1, )).item() if self.kind == "train" else idx
        xm, y = self.dataset._getitem_triplet(idx)
        xm = self.flip_transform(xm)
        x, m = xm[:3], xm[3:]
        x = self.color_transform(x)
        p = self.get_fixed_point(m) if m.sum() > 0.0 else torch.zeros_like(m)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        components = 0
        hot_labels = 0

        # assignments
        composites[:self.fixate] = p + (natural_noise(self.h, self.w) if self.natural else 0.0)
        masks[:self.fixate] = p
        composites[self.fixate:] = x
        masks[self.fixate:] = m
        labels[:] = y

        # adding noise and clamping 
        composites += self.noise * torch.rand_like(composites)
        composites = torch.clamp(composites, 0.0, 1.0)
        masks = 2.0 * (masks - 0.5)

        return composites, labels, masks, components, hot_labels


class ExpSearch_COCO_v2(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 n_iter: int,
                 noise: float = 0.25,):
        
        super().__init__()
        self.kind = "train"
        self.dataset = coco_dataset  # COCO datasets
        self.h, self.w = coco_dataset.h, coco_dataset.w
        self.n_iter = n_iter
        self.noise = noise  # noise scale
        self.dataset._get_tk_cls_anns(False)
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.125),
            transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomAutocontrast(p=0.125),
            transforms.RandomInvert(p=0.125),
            ])

    def build_valid_test(self):
        self.kind = "not_train"
        self.flip_transform = lambda x: x
        self.color_transform = lambda x: x
        self.noise = 0.0

    def __len__(self):
        return self.dataset._len_ if self.kind == "train" else len(self.dataset.tk_cls_anns)

    def __getitem__(self, idx: int):
        idx = torch.randint(0, len(self.dataset.tk_cls_anns), (1, )).item() if self.kind == "train" else idx
        xms, y = self.dataset._getitem_tk_cls_anns(idx)
        xms = self.flip_transform(xms)
        x, ms = xms[:3], xms[3:]
        x = self.color_transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = torch.zeros(self.n_iter, self.dataset.n_classes).float()
        components = 0

        # assignments
        composites[:] = x
        masks[:] = ms.sum(0, keepdim=True)
        labels[:] = y
        hot_labels[:] = one_hot(labels, num_classes=self.dataset.n_classes).float()

        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Search_COCO(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 n_iter: int,
                 noise: float = 0.25,):
        
        super().__init__()
        self.kind = "train"
        self.dataset = coco_dataset  # COCO datasets
        self.h, self.w = coco_dataset.h, coco_dataset.w
        self.n_iter = n_iter
        self.noise = noise  # noise scale
        self.dataset._get_tk_cls_osann(False)
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.125),
            transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomAutocontrast(p=0.125),
            transforms.RandomInvert(p=0.125),
            ])

    def build_valid_test(self):
        self.kind = "not_train"
        self.flip_transform = lambda x: x
        self.color_transform = lambda x: x
        self.noise = 0.0

    def __len__(self):
        return self.dataset._len_ if self.kind == "train" else len(self.dataset.tk_cls_osann)

    def __getitem__(self, idx: int):
        idx = torch.randint(0, len(self.dataset.tk_cls_osann), (1, )).item() if self.kind == "train" else idx
        xm, y = self.dataset._getitem_tk_cls_osann(idx)
        xm = self.flip_transform(xm)
        x, m = xm[:3], xm[3:]
        x = self.color_transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = torch.zeros(self.n_iter, self.dataset.n_classes).float()
        components = 0

        # assignments
        composites[:] = x
        masks[:] = m.sum(0, keepdim=True)
        labels[:] = y
        hot_labels[:] = one_hot(labels, num_classes=self.dataset.n_classes).float()

        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class SearchGrid_COCO(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 bg_dataset: BG20k,
                 n_iter: int,
                 noise: float = 0.25,):
        
        super().__init__()
        self.kind = "train"
        self.dataset = coco_dataset
        self.bg_dataset = bg_dataset
        self.n_bg = len(bg_dataset)
        self.n_iter = n_iter
        self.noise = noise  # noise scale
        self.h, self.w = coco_dataset.h, coco_dataset.w
        self.n_objects = 4
        if self.n_objects == 4:
            self.hh, self.ww = self.h // 2, self.w // 2
            self.grid = {0: (slice(0, self.hh), slice(0, self.ww)), 
                        1: (slice(0, self.hh), slice(self.ww, None)),
                        2: (slice(self.hh, None), slice(0, self.ww)),
                        3: (slice(self.hh, None), slice(self.ww, None))}
        elif self.n_objects == 9:
            self.hh, self.ww = self.h // 3, self.w // 3
            self.grid = {0: (slice(0, self.hh), slice(0, self.ww)), 
                        1: (slice(0, self.hh), slice(self.ww, 2*self.ww)),
                        2: (slice(0, self.hh), slice(2*self.ww, None)),
                        3: (slice(self.hh, 2*self.hh), slice(0, self.ww)),
                        4: (slice(self.hh, 2*self.hh), slice(self.ww, 2*self.ww)),
                        5: (slice(self.hh, 2*self.hh), slice(2*self.ww, None)),
                        6: (slice(2*self.hh, None), slice(0, self.ww)),
                        7: (slice(2*self.hh, None), slice(self.ww, 2*self.ww)),
                        8: (slice(2*self.hh, None), slice(2*self.ww, None))}
        self.dataset._get_triplets()
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.125),
            transforms.ColorJitter(brightness=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomAutocontrast(p=0.125),
            transforms.RandomInvert(p=0.125),
            ])

    def fit_for_grid(self, token: int, ann_id: int):
        ann = self.dataset._get_ann(ann_id)
        x = self.dataset._get_image(token)
        m = self.dataset._get_mask(ann)
        xm = torch.cat((x, m), dim=0)
        xm = self.dataset._crop_box(xm, ann)
        xm = self.dataset._resize_pad(xm, self.hh, self.ww, token, False)
        return xm

    def build_valid_test(self):
        self.kind = "not_train"
        self.flip_transform = lambda x: x
        self.color_transform = lambda x: x
        self.noise = 0.0

    def get_random_digit(self, ind_list: list = None):
        i = random.choice(ind_list)
        return self.dataset.__getitem__(i)

    def rand_crop(self, x, th, tw):
        _, h, w = x.size()
        if h > th and w > tw:
            i = torch.randint(0, h - th + 1, (1, )).item()
            j = torch.randint(0, w - tw + 1, (1, )).item()
            x = x[:, i:i+th, j:j+tw]
        return x

    def make_background(self):
        i = torch.randint(0, self.n_bg, (1, )).item()
        x = self.bg_dataset[i]
        _, h, w = x.shape
        ph, pw = max(0, self.h - h), max(0, self.w - w)
        if ph > 0 or pw > 0:
            x = transforms.functional.pad(x, (1 + pw//2, 1 + ph//2), padding_mode='reflect')
        _, h, w = x.shape
        if h > self.h and w > self.w:
            s = min(h/self.h, w/self.w)
            x = transforms.functional.resize(x, (int(h/s)+1, int(w/s)+1), antialias=False)
        return x[:, :self.h, :self.w]

    def __len__(self):
        return self.dataset._len_ if self.kind == "train" else len(self.dataset.triplets)

    def __getitem__(self, idx: int):
        idx = torch.randint(0, len(self.dataset.triplets), (1, )).item() if self.kind == "train" else idx
        token, ann_id, y = self.dataset.triplets[idx]
        xm = self.fit_for_grid(token, ann_id)
        xm = self.flip_transform(xm)
        x, m = xm[:3], xm[3:]
        # x = self.color_transform(x)
        x *= m

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = torch.zeros(self.n_iter, self.dataset.n_classes).float()
        pos = list(range(self.n_objects))
        random.shuffle(pos)

        # background
        background = self.make_background()
        composites[:] = background

        # assignments
        labels[:] = y
        hot_labels[:] = one_hot(labels[0], self.dataset.n_classes).squeeze().float()
        hs, ws = self.grid[pos[-1]]
        composites[:, :, hs, ws] = x + composites[:, :, hs, ws] * (1.0 - m)
        masks[:, :, hs, ws] = m

        # composing while avoiding overlap
        y_list = random.sample(self.dataset.classes, k=self.n_objects)
        y_list.remove(y) if y in y_list else y_list.pop()
        for i, s in enumerate(y_list):
            j = random.choice(self.dataset.triplet_classes[s])
            token, ann_id, y = self.dataset.triplets[j]
            assert y == s
            xm = self.fit_for_grid(token, ann_id)
            xm = self.flip_transform(xm)
            x, m = xm[:3], xm[3:]
            # x = self.color_transform(x)
            x *= m
            hs, ws = self.grid[pos[i]]
            composites[:, :, hs, ws] = x + composites[:, :, hs, ws] * (1.0 - m)
                    
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Recognition_COCO(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 bg_dataset: BG20k,
                 n_iter: int,
                 stride: int,
                 static: bool,
                 blank: bool,
                 noise: float = 0.25,):
        
        super().__init__()
        self.kind = "train"
        self.dataset = coco_dataset
        self.bg_dataset = bg_dataset
        self.n_iter = n_iter
        self.stride = stride
        self.static = static
        self.blank = blank
        self.noise = noise
        self.n_bg = len(bg_dataset)
        self.h, self.w = coco_dataset.h, coco_dataset.w
        self.dataset._get_triplets()
        self.class_weights = self.dataset.class_weights
        self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)
        self.size_transform = transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(0.5, 1.0))
        self.color_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.125),
            transforms.ColorJitter(brightness=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1)),
            transforms.RandomAutocontrast(p=0.125),
            # transforms.RandomInvert(p=0.125),
            ])

    def build_valid_test(self):
        self.kind = "not_train"
        self.flip_transform = lambda x: x
        self.color_transform = lambda x: x
        self.noise = 0.0
        self.bg_dataset.build_valid_test()

    def _prepare(self, token: int, ann_id: int):
        ann = self.dataset._get_ann(ann_id)
        x = self.dataset._get_image(token)
        m = self.dataset._get_mask(ann)
        xm = torch.cat((x, m), dim=0)
        xm = self.dataset._crop_box(xm, ann)
        xm = self.dataset._resize_pad(xm, self.h, self.w, token, False)
        return xm

    def make_foreground(self):
        window_shape = (self.h, self.w + self.stride * self.n_iter)
        pink_fore = pink(window_shape, 2.5, 0.1, torch.cos).unsqueeze(0)
        pink_fore = 1.0 * ((0.5 + pink_fore / 2) > 0.7)
        return pink_fore

    def rand_crop(self, x, th, tw):
        _, h, w = x.size()
        if h > th and w > tw:
            i = torch.randint(0, h - th + 1, (1, )).item()
            j = torch.randint(0, w - tw + 1, (1, )).item()
            x = x[:, i:i+th, j:j+tw]
        return x

    def make_background(self):
        hs, ws = random.randint(1, self.stride), random.randint(1, self.stride)
        nh, nw = (self.n_iter * hs) + self.h, (self.n_iter * ws) + self.w
        i = torch.randint(0, self.n_bg, (1, )).item()
        x = self.bg_dataset[i]
        _, h, w = x.size()
        ph, pw = max(0, nh - h), max(0, nw - w)
        if ph > 0 or pw > 0:
            x = transforms.functional.pad(x, (1 + pw//2, 1 + ph//2), padding_mode='reflect')
        _, h, w = x.size()
        if h > nh and w > nw:
            s = min(h/nh, w/nw)
            x = transforms.functional.resize(x, (int(h/s)+1, int(w/s)+1), antialias=False)
        return x[:, :nh, :nw], (hs, ws)

    def get_random_digit(self, ind_list: list = None):
        i = random.choice(ind_list)
        return self.dataset.__getitem__(i)

    def __len__(self):
        return self.dataset._len_ if self.kind == "train" else len(self.dataset.triplets)

    def __getitem__(self, idx: int):
        idx = torch.randint(0, len(self.dataset.triplets), (1, )).item() if self.kind == "train" else idx
        token, ann_id, y = self.dataset.triplets[idx]
        xm = self._prepare(token, ann_id)
        xm = self.flip_transform(xm)
        xm = self.size_transform(xm) if random.randint(0, 1) == 0 else xm
        x, m = xm[:3], xm[3:]
        x = self.color_transform(x)
        x *= m

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        hot_labels = torch.zeros(self.n_iter, self.dataset.n_classes).float()
        components = 0

        # get background and foreground
        foreground_color = torch.rand(3, 1, 1)
        foreground_color /= (foreground_color.max() + 1e-6)
        background, (hs, ws) = self.make_background()

        # assignments
        labels[:] = y
        hot_labels[:] = one_hot(labels[0], self.dataset.n_classes).squeeze().float()
        masks[:] = m

        # background
        if self.blank:
            composites[:] = x
        else:
            if self.static:
                composites[:] = background[:, :self.h, :self.w] * (1.0 - m) + x
            else:
                d = random.randint(0, 1)
                for i in range(self.n_iter):
                    if d == 0:
                        hslc = slice(i * hs, self.h + i * hs)
                        wslc = slice(i * ws, self.w + i * ws)
                    else:
                        hslc = slice((self.n_iter - i) * hs, self.h + (self.n_iter - i) * hs)
                        wslc = slice((self.n_iter - i) * ws, self.w + (self.n_iter - i) * ws)
                    composites[i] = background[:, hslc, wslc] * (1.0 - m) + x

        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class ABCeleb(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 the_ids: Union[list, tuple],
                 in_dims: tuple,
                 ):
        super().__init__()
        assert len(the_ids) == 1 or len(the_ids) == 2
        self.dataset = dataset
        self.the_ids = the_ids
        self.in_dims = in_dims
        _, self.h, self.w = in_dims
        self.good_ids = self.__goodones_()
        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w), antialias=False),
            transforms.RandomHorizontalFlip(),
            ])
        
    def __goodones_(self):
        good_ids = []
        for i, a in enumerate(self.dataset.attr):
            if len(self.the_ids) == 1:
                good_ids.append(i)
            elif a[self.the_ids[0]] + a[self.the_ids[1]] == 1:
                good_ids.append(i)
            else:
                continue
        return good_ids

    def __len__(self):
        return len(self.good_ids)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.good_ids[idx]]
        x = x[:, 20:-20, :]
        x = self.transform(x)
        return x, y[self.the_ids[0]]
    

class CelebHair(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 n_iter: int,
                 in_dims: tuple = (3, 128, 128),
                 noise: float = 0.125,
                 kind: str = "train",
                 ):
        super().__init__()
        self.the_ids = (8, 9)
        self.dataset = ABCeleb(dataset, self.the_ids, in_dims=in_dims)
        self.n_iter = n_iter
        self.noise = noise if kind == "train" else 0.0
        self.good_ids = self.dataset.good_ids
        self.kind = kind
        
    def build_valid_test(self):
        self.kind = "not_train"
        self.noise = 0.0
    
    def __len__(self):
        return 61440 if self.kind == "train" else len(self.good_ids)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x += self.noise * torch.rand_like(x)
        x = torch.clamp(x, 0.0, 1.0)
        return do_n_it(x, self.n_iter), do_n_it(y.item(), self.n_iter), 0, 0, 0


class CelebGender(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 n_iter: int,
                 in_dims: tuple = (3, 128, 128),
                 noise: float = 0.125,
                 kind: str = "train",
                 ):
        super().__init__()
        self.the_ids = (20, )
        self.dataset = ABCeleb(dataset, self.the_ids, in_dims=in_dims)
        self.n_iter = n_iter
        self.noise = noise if kind == "train" else 0.0
        self.good_ids = self.dataset.good_ids
        self.kind = kind
        
    def build_valid_test(self):
        self.kind = "not_train"
        self.noise = 0.0
    
    def __len__(self):
        return 61440 if self.kind == "train" else len(self.good_ids)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x += self.noise * torch.rand_like(x)
        x = torch.clamp(x, 0.0, 1.0)
        return do_n_it(x, self.n_iter), do_n_it(y.item(), self.n_iter), 0, 0, 0
    

class CelebSmile(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 n_iter: int,
                 in_dims: tuple = (3, 128, 128),
                 noise: float = 0.125,
                 kind: str = "train",
                 ):
        super().__init__()
        self.the_ids = (31, )
        self.dataset = ABCeleb(dataset, self.the_ids, in_dims=in_dims)
        self.n_iter = n_iter
        self.noise = noise if kind == "train" else 0.0
        self.good_ids = self.dataset.good_ids
        self.kind = kind
        
    def build_valid_test(self):
        self.kind = "not_train"
        self.noise = 0.0
    
    def __len__(self):
        return 61440 if self.kind == "train" else len(self.good_ids)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x += self.noise * torch.rand_like(x)
        x = torch.clamp(x, 0.0, 1.0)
        return do_n_it(x, self.n_iter), do_n_it(y.item(), self.n_iter), 0, 0, 0
    

class CelebGlasses(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 n_iter: int,
                 in_dims: tuple = (3, 128, 128),
                 noise: float = 0.125,
                 kind: str = "train",
                 ):
        super().__init__()
        self.the_ids = (15, )
        self.dataset = ABCeleb(dataset, self.the_ids, in_dims=in_dims)
        self.n_iter = n_iter
        self.noise = noise if kind == "train" else 0.0
        self.good_ids = self.dataset.good_ids
        self.kind = kind
        
    def build_valid_test(self):
        self.kind = "not_train"
        self.noise = 0.0
    
    def __len__(self):
        return 61440 if self.kind == "train" else len(self.good_ids)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x += self.noise * torch.rand_like(x)
        x = torch.clamp(x, 0.0, 1.0)
        return do_n_it(x, self.n_iter), do_n_it(y.item(), self.n_iter), 0, 0, 0
    

class CurveTracing(Dataset):
    def __init__(self,
                 n_samples: int,
                 fix_attend_saccade: tuple,
                 ydim: int,
                 xdim: int,
                 padding: int,
                 resolution: int = 100,
                 noise: float = 0.25,
                 training: bool = True):
        super().__init__()
        self.kernel = 3
        self.bezier_order = 2
        self.n_samples = n_samples  # number of samples
        self.n_fixate, self.n_stimulus, self.n_saccade = fix_attend_saccade
        self.n_iter = sum(fix_attend_saccade)
        self.pad = padding
        self.resolution = resolution
        self.noise = noise
        self.training = training
        self.ydim, self.height = ydim, ydim + 2 * self.pad
        self.xdim, self.width = xdim, xdim + 2 * self.pad
        self.stimulus_kernel = torch.ones(1, 1, self.kernel, self.kernel)
        self.fixate_kernel = torch.ones(1, 1, self.kernel + 2, self.kernel + 2)
        self.saccade_kernel = torch.ones(1, 1, self.kernel + 2, self.kernel + 2)
        self.samples = None
        self.generate_bezier_samples()

    def build_valid_test(self):
        self.noise = 0.0

    def generate_bezier_samples(self):
        self.samples = torch.zeros(self.n_samples, 3, 1, self.height, self.width)
        t_ = torch.linspace(0., 1.0, self.resolution)  # parameter
        c_ = torch.rand(self.n_samples, self.bezier_order + 1, 2, 1)  # coordinates
        for i in range(self.n_samples):
            im_ = torch.zeros(3, 1, self.ydim, self.xdim)
            b_ = bezier_generator(c_[i], t_)  # Bezier curve
            b_h, b_w = (b_[0] * self.ydim).int().tolist(), (b_[1] * self.xdim).int().tolist()
            im_[0, :, b_h, b_w] = 1.0
            im_[1, :, b_h[0], b_w[0]] = 1.0
            im_[2, :, b_h[-1], b_w[-1]] = 1.0
            im_[0] = (conv2d(im_[0], self.stimulus_kernel, padding='same') > 0.0)
            im_[1] = (conv2d(im_[1], self.fixate_kernel, padding='same') > 0.0)
            im_[2] = (conv2d(im_[2], self.saccade_kernel, padding='same') > 0.0)
            self.samples[i] = torch.nn.functional.pad(im_, [self.pad]*4, mode='constant', value=0.0)
        return True

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        target = self.samples[i]
        distractor = self.samples[torch.randint(0, self.n_samples, (1,)).item()]
        if random.choices([True, False]):
            target = transforms.functional.rotate(target, random.choice((0, 90, 180, 270)))
        if random.choices([True, False]):
            distractor = transforms.functional.rotate(distractor, random.choice((0, 90, 180, 270)))

        target_fixation, target_saccade = torch.randperm(2) + 1
        distractor_fixation, distractor_saccade = torch.randperm(2) + 1

        # pre-allocation
        target_composites = torch.zeros(self.n_iter, 1, self.height, self.width)
        distractor_composites = torch.zeros(self.n_iter, 1, self.height, self.width)
        rec_fields = torch.zeros(1, self.height, self.width)
        masks = torch.zeros(self.n_iter, 1, self.height, self.width)
        components = torch.zeros(6, 1, self.height, self.width)

        # receptive field
        rec_fields = target[0] * (1.0 - 1.0 * ((target[1] + target[2] + distractor.sum(dim=0)) > 0.0))

        # building composite
        # target_composites[:self.n_fixate, 1:] += target[target_fixation]  # fixation point
        target_composites[:self.n_fixate] += target[target_fixation]  # fixation point
        target_composites[self.n_fixate:-self.n_saccade] += (target[0] + distractor[0])  # stimulus
        # target_composites[-self.n_saccade:, 1:] += target[target_saccade] + distractor[distractor_saccade] # saccade points
        target_composites[-self.n_saccade:] += target[target_saccade] + distractor[distractor_saccade] # saccade points
        target_composites = torch.clamp(target_composites, 0.0, 1.0)

        # distractor_composites[:self.n_fixate, 1:] += distractor[distractor_fixation]  # fixation point
        distractor_composites[:self.n_fixate] += distractor[distractor_fixation]  # fixation point
        distractor_composites[self.n_fixate:-self.n_saccade] += (target[0] + distractor[0])  # stimulus
        # distractor_composites[-self.n_saccade:, 1:] += target[target_saccade] + distractor[distractor_saccade]  # circles
        distractor_composites[-self.n_saccade:] += target[target_saccade] + distractor[distractor_saccade]  # circles
        distractor_composites = torch.clamp(distractor_composites, 0.0, 1.0)

        # building masks
        masks[:self.n_fixate] += target[target_fixation]
        masks[self.n_fixate:-self.n_saccade] += target[0]
        masks[-self.n_saccade:] += target[target_saccade]
        masks = torch.clamp(masks, 0.0, 1.0)
        masks = 2.0 * (masks - 0.5)

        # noise
        target_composites += self.noise * torch.rand(()) * torch.randn_like(target_composites)
        target_composites = torch.clamp(target_composites, 0.0, 1.0)
        distractor_composites += self.noise * torch.rand(()) * torch.randn_like(distractor_composites)
        distractor_composites = torch.clamp(distractor_composites, 0.0, 1.0)

        # components
        components[0] = target[target_fixation]
        components[1] = target[0]
        components[2] = target[target_saccade]
        components[3] = distractor[distractor_fixation]
        components[4] = distractor[0]
        components[5] = distractor[distractor_saccade]

        if self.training:
            return target_composites, 0, masks, components, 0 
        else:
            return target_composites, distractor_composites, masks, rec_fields, components


class ArrowCur_DS(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_iter: int,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 directory: str = r"./data/",  # directory of the arrow images
                 ):
        
        super().__init__()
        assert n_iter >= 3
        self.directory = os.path.join(directory, "arrows")
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.noise = noise
        self.pad = 2
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = 96, 96
        self.transform = transforms.Pad(self.pad)
        self.n_classes = 10
        self.class_ids = get_classes(self.dataset, 10)
        self.arrows = self.get_arrows()
        self.arrow_pos = [5, 0, 4,
                          2,    1,
                          7, 3, 6]
        self.digit_pos = [(0,  0), (0,  32), (0,  64),
                          (32, 0),           (32, 64),
                          (64, 0), (64, 32), (64, 64)]

    def rand_sample(self, y: int, exclude: bool = False):
        if exclude:
            t = list(range(10))
            t.remove(y)
            cls = random.choice(t)
        else:
            cls = y
        i = self.class_ids[cls][torch.randint(0, len(self.class_ids[cls]), (1, )).item()]
        return self.dataset.__getitem__(i)

    def get_arrows(self):
        arrows = torch.zeros(8, 1, 32, 32)
        for i, file in enumerate(["n", "e", "w", "s", "ne", "nw", "se", "sw"]):
            arrows[i, :, 1:-1, 1:-1] = 1.0 * (transforms.ToTensor()(PILImage.open(os.path.join(self.directory, f"{file}.png")))[0] > 0.0)
        return arrows

    def build_valid_test(self):
        self.noise = 0.0

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        x, y = self.dataset.__getitem__(idx)
        x = self.transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(2, 3, self.h, self.w)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        rgbs = torch.rand(9, 3, 1, 1)
        labels[:] = y
        hot_labels = 0
        b_rgb = torch.rand(3, 1, 1) * 0.5

        t = random.choice(range(8))
        arrow_i, pos_ij = self.arrow_pos[t], self.digit_pos[t]
        components[0, :, 32:64, 32:64] = self.arrows[arrow_i] * rgbs[0]
        components[1, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[1]
        masks[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] += x
        j = 2
        for i in range(8):
            if i != t:
                x, _ = self.rand_sample(y, exclude=True)
                x = self.transform(x)
                pos_ij = self.digit_pos[i]
                components[1, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[j]
                j += 1
        
        composites[0] = (components[0]) + (1.0 - components[0]) * b_rgb
        composites[1:-1] = (components[0] + components[1]) + (1.0 - (components[0] + components[1])) * b_rgb
        composites[-1] = components[1] + (1.0 - components[1]) * b_rgb
        # adding noise and clamping
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Colors:
    def __init__(self, noise: float = 0.0, ext: bool = False) -> None:
        self.noise = noise
        self.ext = ext
        colors = torch.tensor([
            [1.0, 0.0, 0.0,],  # red
            [0.0, 1.0, 0.0,],  # green
            [0.0, 0.0, 1.0,],  # blue
            [0.7, 0.7, 0.0,],  # yellow
            [0.0, 0.7, 0.7,],  # cyan
            [0.7, 0.0, 0.7,],  # magenta
            # [0.57, 0.57, 0.57,],  # white
            # [0.01, 0.01, 0.01,],  # black
        ]).reshape(-1, 3, 1, 1)
        # self.colors = colors if self.ext else colors[:-2]
        self.colors = colors
        self.n_colors = self.colors.shape[0]
        
    def __len__(self):
        return self.n_colors
    
    def __getitem__(self, i):
        c = self.colors[i]
        n = torch.randn_like(c) * self.noise * torch.rand(())
        return (c + n).clamp(0.0, 1.0)


class Textures:
    def __init__(self, h: int, w: int, ext: bool = False):
        self.h, self.w = h, w
        self.ext = ext
        self.n_textures = 4 if self.ext else 3

    def repetitive(self, n: int = 5):
        x = 1.0 * (torch.rand(1, n, n) < 0.5)
        x = x.repeat(1, n+self.h//n, n+self.w//n)[:, :self.h, :self.w]
        return x

    def salt_pepper(self, p: float = 0.5):
        return 1.0 * (torch.rand(1, self.h, self.w) < p)

    def pink(self):
        x = pink((self.h, self.w), c=2.5, a=0.5, actfun=torch.cos)
        x = 0.5 + 0.5 * x
        return x

    def solid(self):
        return torch.rand(3, 1, 1) * torch.ones(1, self.h, self.w)
        # return torch.ones(1, self.h, self.w)

    def __len__(self):
        return self.n_textures

    def __getitem__(self, i):
        return self.salt_pepper() if i == 0 else self.repetitive() if i == 1 else self.pink() if i == 2 else self.solid()


class Recognition_MM(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_grid: int = 4,  # image size
                 ):
        
        super().__init__()
        self.ti = 0.5 # texture intensity
        self.n_mods = 3
        self.dh, self.dw = 32, 32  # digit size
        self.digits = mnist_dataset
        self.colors = Colors(0.25)
        self.textures = Textures(self.dh, self.dw)
        self.n_grid = n_grid
        self.n_iter = 1
        self.n_colors = len(self.colors)
        self.n_textures = len(self.textures)
        self.h, self.w = self.n_grid * self.dh, self.n_grid * self.dw
        self.transform = transforms.Compose([
            transforms.Pad(2),
            transforms.RandomRotation(15)
        ])

    def __len__(self):
        return self.digits.__len__()

    def __getitem__(self, d: int):
        digit, y = self.digits.__getitem__(d)  # digit and label
        digit = self.transform(digit)
        c = torch.randint(0, self.n_colors, (1, )).item()  # color
        t = torch.randint(0, self.n_textures, (1, )).item()  # texture
        x = self.ti * self.textures[t] * (1.0 - digit) + self.colors[c] * digit  # composite
        i, j = torch.randint(0, self.n_grid, (2, ))
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter, self.n_mods).long()  # (n_iter, n_modalities)
        hot_labels = 0

        composites[:, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = x
        masks[0, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = digit
        masks = 2.0 * (masks - 0.5)
        labels[:, 0], labels[:, 1], labels[:, 2] = y, c, t

        return composites, labels, masks, components, hot_labels


class Search_MM(Dataset):
    def __init__(self,
                 mnist_dataset: Dataset,  # MNIST datasets
                 n_grid: int = 4,  # image size
                 n_iter: int = 3,  # number of iterations
                 noise: float = 0.1,
                 scase: int = -1,
                 ):
        
        super().__init__()
        self.train = True
        self.ti = 0.5 # texture intensity
        self.n_mods = 3  # number of modalities
        self.len_mods = (10, 6, 3)  # number of classes per modality
        self.dh, self.dw = 32, 32  # digit size
        self.digits = mnist_dataset
        self.colors = Colors(0.25)
        self.textures = Textures(self.dh, self.dw)
        self.n_grid = n_grid
        self.n_iter = n_iter
        self.noise = noise
        self.scase = scase
        self.n_colors = len(self.colors)
        self.n_textures = len(self.textures)
        self.h, self.w = self.n_grid * self.dh, self.n_grid * self.dw
        self.transform = transforms.Compose([
            transforms.Pad(2),
            transforms.RandomRotation(15)
        ])
        self.class_ids = get_classes(self.digits, 10)

    def get_dyct(self, idx: int, c: int, t: int):
        digit, y = self.digits.__getitem__(idx)
        digit = self.transform(digit)
        return digit, y, self.colors[c], self.textures[t]

    def build_valid_test(self):
        self.noise = 0.0
        self.train = False

    def __len__(self):
        return self.digits.__len__()

    def __getitem__(self, d: int):
        t_y = torch.randint(0, 10, (1, ))
        t_c = torch.randint(0, self.n_colors, (1, ))
        t_t = torch.randint(0, self.n_textures, (1, )) 

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(self.n_grid, self.n_grid, self.n_mods).long()
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        # labels = torch.zeros(self.n_iter, self.n_mods).long()  # (n_iter, n_modalities)
        hot_labels = torch.zeros(self.n_iter, sum(self.len_mods)).float()
        # labels[:, 0], labels[:, 1], labels[:, 2] = t_y, t_c, t_t
        labels = 0
        hot_labels[:, t_y] = 1.0
        hot_labels[:, self.len_mods[0] + t_c] = 1.0
        hot_labels[:, self.len_mods[0] + self.len_mods[1] + t_t] = 1.0

        counter = 0
        if self.scase == -1:
            scase = 0 if torch.rand(1) < 0.1 else 1 if torch.rand(1) < 0.8 else 2 if torch.rand(1) < 0.9 else 3
        else:
            scase = self.scase
        # rand_i, rand_j = torch.randperm(self.n_grid), torch.randperm(self.n_grid)
        rand_ij = torch.randperm(self.n_grid * self.n_grid)
        for ij in rand_ij:
            i, j = ij // self.n_grid, ij % self.n_grid
            if counter < scase:
                d = random.choice(self.class_ids[t_y])
                c, t = t_c, t_t
                counter += 1
            else:
                d = torch.randint(0, len(self.digits), (1, )).item()
                c = torch.randint(0, len(self.colors), (1, )).item()  # color
                t = torch.randint(0, len(self.textures), (1, )).item()  # texture
            digit, y, color, texture = self.get_dyct(d, c, t)
            x = self.ti * texture * (1.0 - digit) + color * digit  # composite
            composites[:, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = x
            components[i, j, 0], components[i, j, 1], components[i, j, 2] = y, c, t
            if y == t_y and c == t_c and t == t_t:
                masks[:, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = 1.0
            # if torch.rand(()) < 0.125:
            #     break
        composites, masks = routine_01(composites, masks, self.noise)
        return composites, labels, masks, components, hot_labels


class Shapes(Dataset):
    def __init__(self,
                 directory: str,
                 height: int = 64,
                 width: int = 64,
                 pre_pad: int = 18,
                 post_pad: int = 0,
                 ext: bool = False,
                 ):
        super().__init__()
        self.ext = ext
        self.height, self.width, self.pre_pad, self.post_pad = height, width, pre_pad, post_pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pre_pad) if self.pre_pad > 0 else lambda x: x,
            transforms.Resize((self.height, self.width), antialias=True),
            transforms.Pad(self.post_pad) if self.post_pad > 0 else lambda x: x,
        ])
        self.shape_names = ["trg.png", "sqr.png", "circ.png", "cross.png", "hex.png", "star.png", "heart.png", "david.png", "crs.png"]
        if self.ext:
            self.shape_names.append("thu.png")
            self.shape_names.append("wu.png")
        self.raw_shapes = self.load_shapes(directory)

    def load_shapes(self, directory):

        raw_shapes = []
        for file in self.shape_names:
            x = 1.0 * (transforms.ToTensor()(PILImage.open(os.path.join(directory, file)))[0])
            raw_shapes.append(self.transform(x.unsqueeze(0)))
        return raw_shapes
    
    def __len__(self):
        return len(self.raw_shapes)
    
    def __getitem__(self, idx):
        return self.raw_shapes[idx]


class ShapeSearch_MM(Dataset):
    def __init__(self,
                 n_grid: int = 4,  # image size
                 n_iter: int = 3,  # number of iterations
                 noise: float = 0.25,  # noise level
                 directory: str = r"./data",  # directory of shapes
                 hard: bool = False,
                 ):
        
        super().__init__()
        self.train = True
        self.directory = os.path.join(directory, "shapes")
        self.n_mods = 3  # number of modalities
        self.dh, self.dw = 64, 64  # shape size
        self.shapes = Shapes(self.directory, self.dh, self.dw)
        self.colors = Colors(0.25)
        self.textures = Textures(self.dh, self.dw)
        self.n_grid = n_grid
        self.n_iter = n_iter
        self.noise = noise
        self.hard = hard
        self.n_shapes = len(self.shapes)
        self.n_colors = len(self.colors)
        self.n_textures = len(self.textures)
        self.len_mods = (self.n_shapes, self.n_colors, self.n_textures)  # number of classes per modality
        self.n_classes = sum(self.len_mods)
        self.h, self.w = self.n_grid * self.dh, self.n_grid * self.dw
        self.transform = transforms.RandomRotation(360)

    def get_dyct(self, i, c, t):
        s_ = self.shapes[i]
        t_ = self.textures[t]
        c_ = self.colors[c]
        m = x = self.transform(s_)
        x = x * t_
        x = x * c_
        return x, m, s_, c_, t_

    def pick_target(self):
        t_s = torch.randint(0, self.n_shapes, (1, ))
        t_c = torch.randint(0, self.n_colors, (1, ))
        t_t = torch.randint(0, self.n_textures, (1, ))
        if self.train and t_s == 0 and t_c == 0 and t_t == 0: 
            r = torch.randint(0, 3, (1, ))
            if r == 0:
                t_s = torch.randint(1, self.n_shapes, (1, ))
            elif r == 1:
                t_c = torch.randint(1, self.n_colors, (1, ))
            else:
                t_t = torch.randint(1, self.n_textures, (1, ))
        elif self.hard and not self.train:
            t_s, t_c, t_t = 0, 0, 0
        return t_s, t_c, t_t

    def build_valid_test(self):
        self.noise = 0.0
        self.colors.noise = 0.0
        self.train = False
        self.transform = lambda x: x

    def __len__(self):
        return (self.n_classes * 1024) if self.train else (self.n_classes * 128)

    def __getitem__(self, d: int):
        t_s, t_c, t_t = self.pick_target()

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = torch.zeros(self.n_grid, self.n_grid, self.n_mods).long()
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        hot_labels = torch.zeros(self.n_iter, sum(self.len_mods)).float()
        labels = 0
        hot_labels[:, t_s] = 1.0
        hot_labels[:, self.len_mods[0] + t_c] = 1.0
        hot_labels[:, self.len_mods[0] + self.len_mods[1] + t_t] = 1.0

        counter = 0
        scase = 0 if torch.rand(1) < 0.1 else 1 if torch.rand(1) < 0.8 else 2 if torch.rand(1) < 0.9 else 3
        # rand_i, rand_j = torch.randperm(self.n_grid), torch.randperm(self.n_grid)
        rand_ij = torch.randperm(self.n_grid * self.n_grid)
        for ij in rand_ij:
            i, j = ij // self.n_grid, ij % self.n_grid
            if counter < scase:
                d, c, t = t_s, t_c, t_t
            else:
                d = torch.randint(0, len(self.shapes), (1, )).item()
                c = torch.randint(0, len(self.colors), (1, )).item()  # color
                t = torch.randint(0, len(self.textures), (1, )).item()  # texture
            counter += 1
            x, m, shape, color, texture = self.get_dyct(d, c, t)
            composites[:, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = x
            components[i, j, 0], components[i, j, 1], components[i, j, 2] = d, c, t
            if d == t_s and c == t_c and t == t_t:
                masks[:, :, i*self.dh:(i+1)*self.dh, j*self.dw:(j+1)*self.dw] = m
        
        composites, masks = routine_01(composites, masks, self.noise)
        return composites, labels, masks, components, hot_labels


class ShapeRecognition_MM(Dataset):
    def __init__(self,
                 n_grid: int = 4,  # image size
                 n_iter: int = 3,  # number of iterations
                 noise: float = 0.25,  # noise level
                 directory: str = r"./data",  # directory of shapes
                 hard: bool = False,
                 ):
        super().__init__()

        self.train = True
        self.directory = os.path.join(directory, "shapes")
        self.n_mods = 3  # number of modalities
        self.dh, self.dw = 64, 64  # shape size
        self.shapes = Shapes(self.directory, self.dh, self.dw)
        self.colors = Colors(0.25)
        self.textures = Textures(self.dh, self.dw)
        self.n_grid = n_grid
        self.n_iter = n_iter
        self.noise = noise
        self.hard = hard
        self.n_shapes = len(self.shapes)
        self.n_colors = len(self.colors)
        self.n_textures = len(self.textures)
        self.len_mods = (self.n_shapes, self.n_colors, self.n_textures)  # number of classes per modality
        self.n_classes = sum(self.len_mods)
        self.h, self.w = self.n_grid * self.dh, self.n_grid * self.dw
        self.transform = transforms.RandomRotation(360)

    def pick_target(self):
        t_s = torch.randint(0, self.n_shapes, (1, ))
        t_c = torch.randint(0, self.n_colors, (1, ))
        t_t = torch.randint(0, self.n_textures, (1, ))
        if self.train and t_s == 0 and t_c == 0 and t_t == 0: 
            r = torch.randint(0, 3, (1, ))
            if r == 0:
                t_s = torch.randint(1, self.n_shapes, (1, ))
            elif r == 1:
                t_c = torch.randint(1, self.n_colors, (1, ))
            else:
                t_t = torch.randint(1, self.n_textures, (1, ))
        elif self.hard and not self.train:
            t_s, t_c, t_t = 0, 0, 0
        return t_s, t_c, t_t

    def get_dyct(self, i, c, t):
        s_ = self.shapes[i]
        t_ = self.textures[t]
        c_ = self.colors[c]
        m = x = self.transform(s_)
        x = x * t_
        x = x * c_
        return x, m, s_, c_, t_

    def build_valid_test(self):
        self.noise = 0.0
        self.colors.noise = 0.0
        self.train = False
        self.transform = lambda x: x

    def __len__(self):
        return (self.n_classes * 1024) if self.train else (self.n_classes * 128)

    def __getitem__(self, d: int):
        t_s, t_c, t_t = self.pick_target()
        x, m, shape, color, texture = self.get_dyct(t_s, t_c, t_t)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter, self.n_mods).long()  # (n_iter, n_modalities)
        hot_labels = 0
        background = torch.rand(3, self.h, self.w)

        i, j = torch.randint(0, self.h-self.dh, (1, )), torch.randint(0, self.w-self.dw, (1, ))
        h_s, w_s = slice(i, i+self.dh), slice(j, j+self.dw)
        composites[:] = background
        composites[:, :, h_s, w_s] = x + background[:, h_s, w_s] * (1.0 - m)
        masks[:, :, h_s, w_s] = m
        labels[:, 0], labels[:, 1], labels[:, 2] = t_s, t_c, t_t
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels


class Cued_CIFAR(Dataset):
    def __init__(self,
                 cifar_dataset: Dataset,  # CIFAR datasets
                 fix_attend: tuple,  # number of fixate and attend iterations
                 n_grid: int = 3,  # image size
                 noise: float = 0.25,  # noise scale
                 in_dims: tuple = (3, 32, 32)
                 ):
        
        super().__init__()
        self.dataset = cifar_dataset
        self.fixate, self.attend = fix_attend
        self.n_iter = sum(fix_attend)
        self.n_grid = n_grid
        self.noise = noise
        self.k = 5
        self.s = 8.0
        _, self.hh, self.ww = in_dims # image size
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.fold = torch.nn.Fold(output_size=(self.h, self.w), kernel_size=(self.hh, self.ww), stride=(self.hh, self.ww))
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=(0.8, 1.8), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                transforms.RandomAutocontrast(p=0.5),
            ])
        self.blur = transforms.GaussianBlur(self.k, self.s)
        self.cue = gaussian_patch(self.hh, self.ww, 5)
        self.gaussian_grid = blur_edges(self.h, self.w, self.n_grid, self.n_grid, self.s).unsqueeze(0)

    def build_valid_test(self):
        self.transform = lambda x: x
        self.noise = 0.0
    
    def edge_blur(self, x: torch.Tensor, gg: torch.Tensor):
        y = self.blur(x)
        return x * gg + y * (1.0 - gg)

    def get_roll(self, i: int, j: int):
        si = torch.randint(- i * self.hh, (self.n_grid - 1 - i) * self.hh, (1, ))  # shifts
        sj = torch.randint(- j * self.ww, (self.n_grid - 1 - j) * self.ww, (1, ))  # shifts
        return si, sj

    def sample_n_shuffle(self, t: torch.Tensor) -> torch.Tensor:
        n = self.n_grid * self.n_grid
        o = torch.randperm(n)
        z = torch.zeros(3, self.hh, self.ww, n)
        z[:, :, :, 0] = t
        for j in range(1, n):
            x = self.dataset[torch.randint(len(self.dataset), (1, ))][0]
            x = self.transform(x)
            z[:, :, :, j] = x
        z = z[None, :, :, :, o]
        z = z.view(1, 3*self.hh*self.ww, n)
        return self.fold(z)[0], o

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x = self.transform(x)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        components = 0
        hot_labels = 0
        labels[:] = y
        
        z, o = self.sample_n_shuffle(x)
        tar_ij = (o == 0).nonzero().item()
        i, j = tar_ij // self.n_grid, tar_ij % self.n_grid
        i_slice, j_slice = slice(i*self.hh, (i+1)*self.hh), slice(j*self.ww, (j+1)*self.ww)

        composites[:self.fixate, :, i_slice, j_slice] = self.cue
        composites[self.fixate:] = z
        masks[:self.fixate, :, i_slice, j_slice] = self.cue
        masks[self.fixate:, :, i_slice, j_slice] = 1.0
            
        si, sj = self.get_roll(i, j)
        composites[:] = torch.roll(composites[:], shifts=(si, sj), dims=(-2, -1))
        masks[:] = torch.roll(masks[:], shifts=(si, sj), dims=(-2, -1))
        gg = torch.roll(self.gaussian_grid, shifts=(si, sj), dims=(-2, -1))
        composites[self.fixate:] = self.edge_blur(composites[self.fixate:], gg)
        composites, masks = routine_01(composites, masks, self.noise)
        return composites, labels, masks, components, hot_labels


class Single_CIFAR(Dataset):
    def __init__(self,
                 cifar_dataset: Dataset,  # CIFAR datasets
                 n_iter: int,  # number of fixate and attend iterations
                 n_grid: int = 3,  # image size
                 noise: float = 0.25,  # noise scale
                 in_dims: tuple = (3, 32, 32),
                 centered: bool = False,
                 background: bool = False,
                 ):
        
        super().__init__()
        self.dataset = cifar_dataset
        self.n_iter = n_iter
        self.n_grid = n_grid
        self.noise = noise
        self.centered = centered
        self.background = background
        _, self.hh, self.ww = in_dims # image size
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=(0.8, 1.8), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                transforms.RandomAutocontrast(p=0.5),
            ])

    def build_valid_test(self):
        self.transform = lambda x: x
        self.noise = 0.0
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x = self.transform(x)

        # pre-allocation
        composites = torch.zeros(max(self.n_iter, 1), 3, self.h, self.w)
        labels = torch.zeros(max(self.n_iter, 1)).long()
        masks = 0
        components = 0
        hot_labels = 0
        labels[:] = y

        if self.background:
            composites[:] = natural_noise(self.h, self.w)
        if self.centered:
            i, j = (self.h - self.hh)//2, (self.w - self.ww)//2
        else:
            i = torch.randint(0, self.h - self.hh, (1, ))
            j = torch.randint(0, self.w - self.ww, (1, ))
        composites[:, :, i:i+self.hh, j:j+self.ww] = x

        composites += torch.rand(()) * self.noise * torch.rand_like(composites)
        composites = torch.clamp(composites, 0.0, 1.0)
        if self.n_iter == 0:
            return composites[0], labels[0]
        else:
            return composites, labels, masks, components, hot_labels


class Scattered_CIFAR(Dataset):
    def __init__(self,
                 cifar_dataset: Dataset,  # CIFAR datasets
                 n_iter: int,  # number of fixate and attend iterations
                 n_grid: int = 3,  # image size
                 n_pieces: int = 2,  # number of pieces
                 noise: float = 0.25,  # noise scale
                 in_dims: tuple = (3, 32, 32),
                 hard: bool = False,
                 separate: bool = True,
                 ):
        
        super().__init__()
        self.train = True
        self.dataset = cifar_dataset
        self.n_iter = n_iter
        self.n_grid = n_grid
        self.noise = noise
        self.n_pieces = n_pieces
        self.hard = hard
        self.separate = separate
        self.png = self.n_grid * self.n_pieces
        _, self.hh, self.ww = in_dims  # image size
        assert self.hh%self.n_pieces == 0 and self.ww%self.n_pieces == 0, 'Image size must be divisible by n_pieces!'
        self.zh, self.zw = self.hh//self.n_pieces, self.ww//self.n_pieces
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.fold = torch.nn.Fold(output_size=(self.h, self.w), kernel_size=(self.zh, self.zw), stride=(self.zh, self.zw))
        self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=(0.8, 1.8), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                transforms.RandomAutocontrast(p=0.5),
            ])
        
    def build_valid_test(self):
        self.train = False
        self.trans = lambda x: x
        self.noise = 0.0

    def scatter(self) -> torch.Tensor:
        n = self.n_grid * self.n_grid
        s = self.n_pieces * self.n_pieces
        z = torch.zeros(3, self.zh, self.zw, n * s)
        k, _ = self.dataset[torch.randint(len(self.dataset), (1, ))]
        for j in range(n):
            x = k if self.hard else self.dataset[torch.randint(len(self.dataset), (1, ))][0]
            x = self.trans(x)
            z[:, :, :, s*j:s*(j+1)] = x.unfold(1, self.zh, self.zh).unfold(2, self.zw, self.zw).reshape(3, -1, self.zh, self.zw).permute(0, 2, 3, 1)
        z = z[None, :, :, :, torch.randperm(n * s)]
        z = z.view(1, 3*self.zh*self.zw, n * s)
        return self.fold(z)[0]

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(2):
            for j in range(2):
                sh = slice(i*self.zh, (i+1)*self.zh)
                sw = slice(j*self.zw, (j+1)*self.zw)
                x[:, sh, sw] = self.trans(x[:, sh, sw])
        return x

    def draw_grid_lines(self, x: torch.Tensor):
        n = natural_noise(self.h, self.w)
        x[:, torch.arange(0, self.png*self.zh, self.zh)] = n[:, torch.arange(0, self.png*self.zh, self.zh)]
        x[:, :, torch.arange(0, self.png*self.zw, self.zw)] = n[:, :, torch.arange(0, self.png*self.zw, self.zw)]
        return x
    
    def get_roll(self, i: int, j: int):
        si = torch.randint(- i * self.zh, (self.png - self.n_pieces - i) * self.zh, (1, ))  # shifts
        sj = torch.randint(- j * self.zw, (self.png - self.n_pieces - j) * self.zw, (1, ))  # shifts
        return si, sj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        # pre-allocation
        composites = torch.zeros(max(self.n_iter, 1), 3, self.h, self.w)
        labels = torch.zeros(max(self.n_iter, 1)).long()
        masks = 0
        components = 0
        hot_labels = 0

        labels[:] = y
        x = self.trans(x)

        z = self.scatter()
        i, j = torch.randint(0, self.png - self.n_pieces, (2, ))
        z[:, i*self.zh:(i+self.n_pieces)*self.zh, j*self.zw:(j+self.n_pieces)*self.zw] = x

        z = self.draw_grid_lines(z) if (self.train and self.separate) else z
        si, sj = self.get_roll(i, j)
        composites[:] = torch.roll(z, shifts=(si, sj), dims=(-2, -1))
        composites += torch.rand(1) * self.noise * torch.rand_like(composites)
        composites = torch.clamp(composites, 0.0, 1.0)

        if self.n_iter == 0:
            return composites[0], labels[0]
        else:
            return composites, labels, masks, components, hot_labels


class Cued_Scattered_CIFAR(Dataset):
    def __init__(self,
                 cifar_dataset: Dataset,  # CIFAR datasets
                 fix_attend: tuple,  # number of fixate and attend iterations
                 n_grid: int = 3,  # image size
                 noise: float = 0.25,  # noise scale
                 ):
        
        super().__init__()
        self.dataset = cifar_dataset
        self.fixate, self.attend = fix_attend
        self.n_iter = sum(fix_attend)
        self.n_grid = n_grid
        self.noise = noise
        self.zh, self.zw = 16, 16
        self.hh, self.ww = 32, 32  # image size
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.trans = transforms.Compose([
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=(0.8, 1.8), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                transforms.RandomAutocontrast(p=0.5),
            ])
        self.cue = gaussian_patch(self.hh, self.ww, 5)

    def scatter(self) -> torch.Tensor:
        n = (self.n_grid * self.n_grid) - 1
        z = torch.zeros(4 * n, 3, self.zh, self.zw)
        for k in range(n):
            x, _ = self.dataset[torch.randint(len(self.dataset), (1,))]
            for i in range(2):
                for j in range(2):
                    sh = slice(i*self.zh, (i+1)*self.zh)
                    sw = slice(j*self.zw, (j+1)*self.zw)
                    z[j + 2 * i + 4 * k] = self.trans(x[:, sh, sw])
        z = z[torch.randperm(4 * n)]
        return z

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(2):
            for j in range(2):
                sh = slice(i*self.zh, (i+1)*self.zh)
                sw = slice(j*self.zw, (j+1)*self.zw)
                x[:, sh, sw] = self.trans(x[:, sh, sw])
        return x

    def build_valid_test(self):
        self.flip = lambda x: x
        self.trans = lambda x: x
        self.noise = 0.0
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # pre-allocation
        temposite = torch.zeros(3, self.h, self.w)
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        cue = torch.zeros(3, self.h, self.w)
        components = 0
        hot_labels = 0

        x, y = self.dataset[idx]
        labels[:] = y
        x = self.flip(x)

        i, j = torch.randint(0, self.n_grid, (2, ))
        tar_i, tar_j = i, j
        temposite[:, i*self.hh:(i+1)*self.hh, j*self.ww:(j+1)*self.ww] = x
        z = self.scatter()
        k = 0
        for ii in range(2 * self.n_grid):
            for jj in range(2 * self.n_grid):
                if ii not in (2 * i, (2 * i) + 1) or jj not in (2 * j, (2 * j) + 1):
                    temposite[:, ii*self.zh:(ii+1)*self.zh, jj*self.zw:(jj+1)*self.zw] = z[k]
                    k += 1
        if i == 0:
            si = torch.randint(0, (self.n_grid - 1) * self.hh, (1, ))  # shifts
        elif i == self.n_grid - 1:
            si = - torch.randint(0, (self.n_grid - 1) * self.hh, (1, ))
        else:
            si = torch.randint(-self.hh, self.hh, (1, ))  # shifts
        if j == 0:
            sj = torch.randint(0, (self.n_grid - 1) * self.ww, (1, ))  # shifts
        elif j == self.n_grid - 1:
            sj = - torch.randint(0, (self.n_grid - 1) * self.ww, (1, ))
        else:
            sj = torch.randint(-self.ww, self.ww, (1, ))  # shifts
        
        cue[:, tar_i*self.hh:(tar_i+1)*self.hh, tar_j*self.ww:(tar_j+1)*self.ww] = self.cue
        cue = torch.roll(cue, shifts=(si, sj), dims=(1, 2))
        temposite = torch.roll(temposite, shifts=(si, sj), dims=(1, 2))
        masks[:self.fixate] = cue[0:1]
        masks[self.fixate:, :, i*self.hh+si:(i+1)*self.hh+si, j*self.ww+sj:(j+1)*self.ww+sj] = 1.0
        composites[:self.fixate] = cue
        composites[self.fixate:] = temposite
        composites, masks = routine_01(composites, masks, self.noise)
        temposite, x, z, cue = None, None, None, None

        return composites, labels, masks, components, hot_labels


class Search_CIFAR(Dataset):
    def __init__(self,
                 cifar_dataset: Dataset,  # CIFAR datasets
                 n_iter: tuple,  # number of fixate and attend iterations
                 n_grid: int = 3,  # image size
                 noise: float = 0.25,  # noise scale
                 in_dims: tuple = (3, 32, 32),  # image size
                 n_classes: int = 100,  # number of classes
                 ):
        
        super().__init__()
        self.dataset = cifar_dataset
        self.n_iter = n_iter
        self.n_grid = n_grid
        self.noise = noise
        self.in_dims = in_dims
        self.n_classes = n_classes
        self.k = 5
        self.s = 8.0
        _, self.hh, self.ww = in_dims # image size
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=(0.8, 1.8), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                transforms.RandomAutocontrast(p=0.5),
            ])
        self.blur = transforms.GaussianBlur(self.k, self.s)
        self.gaussian_grid = blur_edges(self.h, self.w, self.n_grid, self.n_grid, self.s).unsqueeze(0)

    def build_valid_test(self):
        self.transform = lambda x: x
        self.noise = 0.0
    
    def edge_blur(self, x: torch.Tensor, gg: torch.Tensor):
        y = self.blur(x)
        return x * gg + y * (1.0 - gg)

    def get_roll(self, i: int, j: int):
        si = torch.randint(- i * self.hh, (self.n_grid - 1 - i) * self.hh, (1, ))  # shifts
        sj = torch.randint(- j * self.ww, (self.n_grid - 1 - j) * self.ww, (1, ))  # shifts
        return si, sj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        labels = torch.zeros(self.n_iter).long()
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        components = 0
        hot_labels = torch.zeros(self.n_iter, self.n_classes).float()
        
        set_target = True
        rand_ij = torch.randperm(self.n_grid * self.n_grid)
        for ij in rand_ij:
            i, j = ij // self.n_grid, ij % self.n_grid
            i_slice, j_slice = slice(i*self.hh, (i+1)*self.hh), slice(j*self.ww, (j+1)*self.ww)
            if set_target:
                x, y = self.dataset[idx]
                x = self.transform(x)
                labels[:] = y
                composites[:, :, i_slice, j_slice] = x
                masks[:, :, i_slice, j_slice] = 1.0
                hot_labels[:, y] = 1.0
                set_target = False
                ti, tj = i, j
            else:
                while True:
                    idx = torch.randint(len(self.dataset), (1,))
                    x, y = self.dataset[idx]
                    if y != labels[0]:
                        break
                x = self.transform(x)
                composites[:, :, i_slice, j_slice] = x
    
        composites, masks = routine_01(composites, masks, self.noise)
        return composites, labels, masks, components, hot_labels


class ShapeRecognition_FBG(Dataset):
    def __init__(self,
                 n_iter: int = 3,  # number of iterations
                 directory: str = r"./data",  # directory of shapes
                 noise: float = 0.25,  # noise level
                 hard: bool = False,
                 ext: bool = True,
                 empty: bool = False,
                 invert: bool = False,
                 ):
        super().__init__()

        self.train = True
        self.directory = os.path.join(directory, "shapes")
        self.n_mods = 3  # number of modalities
        self.h, self.w = 128, 128
        self.shapes = Shapes(self.directory, 64, 64, pre_pad=0, post_pad=32)
        self.colors = Colors(0.25)
        self.fg_textures = Textures(self.h, self.w, ext=ext)
        self.bg_textures = Textures(self.h, self.w, ext=ext)
        self.n_iter = n_iter
        self.noise = noise
        self.hard = hard
        self.empty = empty
        self.invert = invert
        self.n_shapes = len(self.shapes)
        self.n_colors = len(self.colors)
        self.n_fg_textures = len(self.fg_textures)
        self.n_bg_textures = len(self.bg_textures)
        self.len_mods = (self.n_shapes, self.n_colors, self.n_fg_textures)  # number of classes per modality
        self.n_classes = sum(self.len_mods)
        self.transform = transforms.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2), scale=(0.7, 1.2))

    def pick_target(self):
        t_s = torch.randint(0, self.n_shapes, (1, ))
        t_c = torch.randint(0, self.n_colors, (1, ))
        t_t = torch.randint(0, self.n_fg_textures, (1, ))
        if self.train and t_s == t_c and t_t == t_s: 
            if t_t.item() == 0:
                t_t = torch.randint(1, self.n_fg_textures, (1, ))
            elif t_c.item() == 1:
                t_c = torch.randint(2, self.n_colors, (1, ))
            else:
                t_s = torch.randint(3, self.n_shapes, (1, ))
        elif self.hard and not self.train:
            r = torch.randint(0, 3, (1, ))
            t_s, t_c, t_t = r, r, r
        return t_s, t_c, t_t

    def pick_background(self, t_c, t_t):
        while True:
            b_c = torch.randint(0, self.n_colors, (1, ))
            b_t = torch.randint(0, self.n_bg_textures, (1, ))
            if b_c != t_c or b_t != t_t:
                break
        t_ = self.fg_textures[b_t]
        c_ = self.colors[b_c]
        return t_ * c_

    def get_dyct(self, i, c, t):
        s_ = self.shapes[i]
        t_ = self.fg_textures[t]
        c_ = self.colors[c]
        m = x = self.transform(s_)
        x = x * t_
        x = x * c_
        return x, m, s_, c_, t_

    def build_valid_test(self):
        self.noise = 0.0
        self.colors.noise = 0.0
        self.train = False
        self.transform = lambda x: x

    def __len__(self):
        return (self.n_classes * 1024) if self.train else (self.n_classes * 128)

    def __getitem__(self, d: int):
        t_s, t_c, t_t = self.pick_target()
        x, m, shape, color, texture = self.get_dyct(t_s, t_c, t_t)

        # pre-allocation
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        components = 0
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter, self.n_mods).long()  # (n_iter, n_modalities)
        hot_labels = 0
        background = self.pick_background(t_c, t_t)

        # composites[:] = background
        composites[:] = (0.0 if self.invert else x) + (background * (1.0 - m) if not self.empty else 0.0)
        masks[:] = m
        labels[:, 0], labels[:, 1], labels[:, 2] = t_s, t_c, t_t
        composites, masks = routine_01(composites, masks, self.noise)

        return composites, labels, masks, components, hot_labels

class PsychCompare(Dataset):
    def __init__(self,
                 episodes: tuple = (1, 2, 1, 3),
                 a_range: tuple = (0.2, 0.8),
                 n_range: tuple = (0.0, 0.2),
                 b_range: tuple = (0.125, 0.25),
                 d_range: tuple = (0.0, 0.2),
                 diff_r: float = 0.9,
                 biased: float = 0.0,
                 cue_loc: int = None,
                 gabor_a_0: float = None,
                 gabor_a_1: float = None,
                 force_range: bool = False,
                 force_label: bool = False,
                 force_cue: bool = False,
                 force_loc: bool = False,
                 force_diff: bool = False,
                 n_samples: int = 1024,
                 ):
        super().__init__()
        self.n_grid = 3
        self.hh, self.ww = 32, 32
        self.offcenter_locations = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.gg = self.n_grid * self.n_grid
        self.epidodes = episodes
        self.ep_no1, self.ep_cue, self.ep_no2, self.ep_stim = episodes  # noise1, cue, noise1, stimulus
        self.ep_no1_cue = self.ep_no1 + self.ep_cue
        self.ep_no1_cue_no2 = self.ep_no1_cue + self.ep_no2
        self.n_iter = sum(episodes)
        self.a_range = a_range
        self.n_range = n_range
        self.b_range = b_range
        self.diff_r = diff_r
        self.biased = biased
        self.cue_loc = cue_loc
        self.gabor_a_0 = gabor_a_0
        self.gabor_a_1 = gabor_a_1
        self.force_range = force_range
        self.force_label = force_label
        self.force_loc = force_loc
        self.force_diff = force_diff
        self.n_samples = n_samples
        self.raw_gabors = self.load_gabors()
        self.gaussian = gaussian_patch(self.hh, self.ww, 6.0)
        self.cue = gaussian_patch(self.hh, self.ww, 6.0)
        self.ranger = lambda x: x[0] + (torch.rand(()) * (x[1] - x[0])) if isinstance(x, (tuple, list)) else x

    def load_gabors(self):
        z = gaussian_patch(64, 64, 12.0)
        gabors = torch.zeros(180, 1, 64, 64)
        for i in range(180):
            x = gabor(64, 64, 0.75, i * math.pi / 180.0, 12.0)
            gabors[i, 0] = x * z
            gabors[i, 0] /= gabors[i, 0].max()  # normalize to [0, 1]
        return transforms.Resize((self.hh, self.ww), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(gabors)

    def make_slice(self, i: int, j: int):
        h_slice = slice(i * self.hh, (i + 1) * self.hh)
        w_slice = slice(j * self.ww, (j + 1) * self.ww)
        return h_slice, w_slice

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        composites = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        components = 0
        hot_labels = 0
        
        gabor_loc = self.offcenter_locations[torch.randperm(8)[:2]]
        gabor_a_0 = self.a_range if self.force_range else self.ranger(self.a_range) # gabor amplitude for the first gabor
        gabor_a_0 = gabor_a_0 if self.gabor_a_0 is None else self.gabor_a_0  # override if provided
        gabor_a_1 = gabor_a_0 - self.diff_r if self.force_diff else self.ranger(self.a_range) # gabor amplitude for the second gabor
        gabor_a_1 = gabor_a_1 if self.gabor_a_1 is None else self.gabor_a_1  # override if provided
        gabor_p_0 = gabor_loc[0]  # gabor position for the first gabor
        gabor_p_1 = gabor_loc[1]  # gabor position for the second gabor
        gabor_r_0 = torch.randint(0, 180, (())).item()  # rotation for the first gabor
        gabor_r_1 = torch.randint(0, 180, (())).item()  # rotation for the second gabor
        target_loc = gabor_p_0 if gabor_a_0 > gabor_a_1 else gabor_p_1
        if self.cue_loc is None:
            if self.force_loc:
                cue_loc = gabor_p_0.item()
            elif torch.rand(()) < self.biased:
                cue_loc = target_loc.item()
            else:
                cue_loc = torch.randint(0, self.gg, ()).item()
        else:
            cue_loc = gabor_p_1 if self.cue_loc == 1 else gabor_p_0 if self.cue_loc == 0 else 4
            cue_loc = cue_loc.item() if isinstance(cue_loc, torch.Tensor) else cue_loc
        # cue_loc = target_loc.item() if (self.force_loc or torch.rand(()) < self.biased) else torch.randint(0, self.gg, (())).item()

        # composites
        i, j = divmod(cue_loc, self.n_grid)
        h_slice, w_slice = self.make_slice(i, j)
        composites[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.cue
        masks[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.gaussian
        labels[self.ep_no1:self.ep_no1_cue] = cue_loc

        for a, p, r in zip([gabor_a_0, gabor_a_1], [gabor_p_0, gabor_p_1], [gabor_r_0, gabor_r_1]):
            i, j = divmod(p.item(), self.n_grid)
            h_slice, w_slice = self.make_slice(i, j)
            gabor = a * self.raw_gabors[r]
            composites[-self.ep_stim:, :, h_slice, w_slice] += gabor

        i, j = divmod(target_loc.item(), self.n_grid)
        h_slice, w_slice = self.make_slice(i, j)
        masks[-self.ep_stim:, :, h_slice, w_slice] = self.gaussian
        labels[-self.ep_stim:] = gabor_p_0 if self.force_label else target_loc

        # # motion and locations
        # show_gabor = True if self.force_label else (torch.rand(()) < 0.5)
        # show_cue = True if self.force_cue else (torch.rand(()) < 0.5)
        # gabor_loc = self.offcenter_locations[torch.randint(0, 8, (()))].item()
        # cue_loc = gabor_loc if (self.force_loc or torch.rand(()) < self.biased) else torch.randint(0, self.gg, (())).item()
        # amplitude = self.a_range if self.force_range else self.ranger(self.a_range)

        masks = 2.0 * (masks - 0.5)
        noise = self.ranger(self.n_range)
        bias = min(self.ranger(self.b_range), 1.0 - max(gabor_a_0, gabor_a_1))
        composites[:] += (noise * torch.randn_like(composites) + bias)
        torch.clamp_(composites, 0.0, 1.0)
        labels = labels.long()
        
        return composites, labels, masks, components, hot_labels



class PsychChange(Dataset):
    def __init__(self,
                 episodes: tuple = (1, 2, 1, 3),
                 r_range: int = 45,
                 r_base: int = 5,
                 biased: float = 0.0,
                 force_range: bool = False,
                 force_label: bool = False,
                 force_neutral: bool = False,
                 n_objects: int = 8,
                 noise: float = 0.25,
                 n_samples: int = 1024,
                 ):
        super().__init__()
        self.n_grid = 3
        self.n_objects = n_objects
        self.hh, self.ww = 32, 32
        self.offcenter_locations = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.gg = self.n_grid * self.n_grid
        self.epidodes = episodes
        self.ep_no1, self.ep_cue, self.ep_no2, self.ep_stim = episodes  # noise1, cue, noise1, stimulus
        self.ep_no1_cue = self.ep_no1 + self.ep_cue
        self.ep_no1_cue_no2 = self.ep_no1_cue + self.ep_no2
        self.n_iter = sum(episodes)
        self.r_range = r_range
        self.r_base = r_base
        self.biased = biased
        self.force_range = force_range
        self.force_label = force_label
        self.force_neutral = force_neutral
        self.noise = noise
        self.n_samples = n_samples
        self.raw_gabors = self.load_gabors()
        self.gaussian = gaussian_patch(self.hh, self.ww, 6.0)
        self.cue = gaussian_patch(self.hh, self.ww, 6.0)
        self.ranger = lambda x: x[0] + (torch.rand(()) * (x[1] - x[0])) if isinstance(x, (tuple, list)) else x
        
    def load_gabors(self):
        z = gaussian_patch(64, 64, 12.0)
        gabors = torch.zeros(180, 1, 64, 64)
        for i in range(180):
            x = gabor(64, 64, 0.75, i * math.pi / 180.0, 12.0)
            gabors[i, 0] = x * z
            gabors[i, 0] /= gabors[i, 0].max()  # normalize to [0, 1]
        return transforms.Resize((self.hh, self.ww), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(gabors)

    def make_slice(self, i: int, j: int):
        h_slice = slice(i * self.hh, (i + 1) * self.hh)
        w_slice = slice(j * self.ww, (j + 1) * self.ww)
        return h_slice, w_slice

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        composites = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter, dtype=torch.long)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        components = 0
        hot_labels = 0
        
        # motion and locations
        n_objects = self.n_objects if self.n_objects > 0 else torch.randint(1, self.gg, (())).item()
        no_change = False if self.force_label else (torch.rand(()) < 0.5)
        locations = self.offcenter_locations[torch.randperm(self.gg-1)].tolist()
        m_noise = torch.randint(-self.r_base, self.r_base, (n_objects, self.ep_stim)) if self.r_base > 0 else torch.zeros(n_objects, self.ep_stim)
        offset = torch.randint(0, 180, (n_objects, 1))
        motions = m_noise + offset
        if no_change:
            labels[-self.ep_stim:] = 4
        else:
            labels[-self.ep_stim:] = locations[0]
            if self.force_range:
                m_delta = self.r_range
            else:
                m_delta = (-1 if torch.rand(()) < 0.5 else 1) * torch.randint(self.r_base, self.r_range, (()))
            motions[0] += (m_delta * torch.arange(self.ep_stim)).long()
        motions = motions % 180

        # building compositions
        if self.ep_no1 > 0:  # noise1
            composites[:self.ep_no1, :, :, :] = self.noise * torch.rand(self.ep_no1, 1, self.h, self.w)
        if self.ep_no2 > 0:  # noise2
            composites[self.ep_no1_cue:self.ep_no1_cue_no2, :, :, :] = self.noise * torch.rand(self.ep_no2, 1, self.h, self.w)

        # cue
        cue_loc = locations[1] if (self.biased < 0.0 and n_objects > 1) else locations[0] if torch.rand(()) < self.biased else torch.randint(0, self.gg, ()).item()
        cue_loc = 4 if self.force_neutral else cue_loc
        i, j = divmod(cue_loc, self.n_grid)
        h_slice, w_slice = self.make_slice(i, j)
        composites[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.cue
        masks[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.gaussian

        # gratings
        for k in range(n_objects):
            loc = locations[k]
            i, j = divmod(loc, self.n_grid)
            h_slice, w_slice = self.make_slice(i, j)
            # print(self.ep_stim, h_slice, w_slice, k, motions, motions[k])
            composites[-self.ep_stim:, :, h_slice, w_slice] = self.raw_gabors[motions[k]]
            if k == 0 and not no_change:
                masks[-self.ep_stim:, :, h_slice, w_slice] = self.gaussian
    
        composites += torch.randn_like(composites) * self.noise * torch.rand(())
        torch.clamp_(composites, 0.0, 1.0)

        return composites, labels, masks, components, hot_labels


class PsychGrating(Dataset):
    def __init__(self,
                 episodes: tuple = (1, 2, 1, 3),
                 a_range: tuple = (0.1, 0.9),
                 n_range: tuple = (0.25, 0.5),
                 b_range: tuple = (0.125, 0.25),
                 biased: float = 0.0,
                 rot_noise: int = 5,
                 cls_rot: bool = False,
                 force_range: bool = False,
                 force_label: bool = False,
                 force_cue: bool = False,
                 force_loc: bool = False,
                 gabor_cue: bool = False,
                 show_cue: bool = False,
                 cue_loc_off: int = False,
                 decay_rate: float = 0.0,
                 n_samples: int = 1024,
                 ):
        super().__init__()
        self.n_grid = 3
        self.hh, self.ww = 32, 32
        self.offcenter_locations = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
        self.h, self.w = self.n_grid * self.hh, self.n_grid * self.ww
        self.gg = self.n_grid * self.n_grid
        self.epidodes = episodes
        self.ep_no1, self.ep_cue, self.ep_no2, self.ep_stim = episodes  # noise1, cue, noise1, stimulus
        self.ep_no1_cue = self.ep_no1 + self.ep_cue
        self.ep_no1_cue_no2 = self.ep_no1_cue + self.ep_no2
        self.n_iter = sum(episodes)
        self.a_range = a_range
        self.n_range = n_range
        self.b_range = b_range
        self.biased = biased
        self.cls_rot = cls_rot
        self.rot_noise = rot_noise
        self.force_range = force_range
        self.force_label = force_label
        self.force_cue = force_cue
        self.force_loc = force_loc
        self.show_cue = show_cue
        self.cue_loc_off = cue_loc_off
        self.gabor_cue = gabor_cue
        self.decay_rate = decay_rate
        self.n_samples = n_samples
        self.raw_gabors = self.load_gabors()
        self.gaussian = gaussian_patch(self.hh, self.ww, 8.0)
        self.cue = gaussian_patch(self.hh, self.ww, 8.0)
        decay = torch.pow(torch.tensor([self.decay_rate]), torch.arange(self.ep_stim))
        self.decay = torch.cat([torch.ones(self.ep_no1_cue_no2), decay], dim=0)
        self.ranger = lambda x: x[0] + (torch.rand(()) * (x[1] - x[0])) if isinstance(x, (tuple, list)) else x

    def load_gabors(self):
        z = gaussian_patch(64, 64, 12.0)
        gabors = torch.zeros(180, 1, 64, 64)
        for i in range(180):
            x = gabor(64, 64, 0.75, i * math.pi / 180.0, 12.0)
            gabors[i, 0] = x * z
            gabors[i, 0] /= gabors[i, 0].max()  # normalize to [0, 1]
        return transforms.Resize((self.hh, self.ww), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(gabors)

    def make_slice(self, i: int, j: int):
        h_slice = slice(i * self.hh, (i + 1) * self.hh)
        w_slice = slice(j * self.ww, (j + 1) * self.ww)
        return h_slice, w_slice

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        composites = torch.zeros(self.n_iter, 1, self.h, self.w)
        labels = torch.zeros(self.n_iter, 2) if self.cls_rot else torch.zeros(self.n_iter)
        masks = torch.zeros(self.n_iter, 1, self.h, self.w)
        components = 0
        hot_labels = 0
        
        # motion and locations
        show_gabor = True if self.force_label else (torch.rand(()) < 0.5)
        show_cue = True if self.force_cue else (torch.rand(()) < 0.5)
        gabor_loc = self.offcenter_locations[torch.randint(0, 8, (()))].item()
        cue_loc = gabor_loc if (self.force_loc or torch.rand(()) < self.biased) else torch.randint(0, self.gg, (())).item()
        cue_loc = self.offcenter_locations[(gabor_loc+4)%8].item() if self.cue_loc_off else cue_loc
        amplitude = self.a_range if self.force_range else self.ranger(self.a_range)

        # cue
        this_cue = self.raw_gabors[torch.randint(0, 180, (())).item()] if self.gabor_cue else self.cue
        if show_cue:
            i, j = divmod(cue_loc, self.n_grid)
            h_slice, w_slice = self.make_slice(i, j)
            composites[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = this_cue
            masks[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.gaussian
            if self.cls_rot:
                labels[self.ep_no1:self.ep_no1_cue, 0] = cue_loc
                # labels[self.ep_no1:self.ep_no1_cue, 1] = rot_label
            else:
                labels[self.ep_no1:self.ep_no1_cue] = cue_loc
        else:
            if self.show_cue:
                i, j = divmod(4, self.n_grid)
                h_slice, w_slice = self.make_slice(i, j)
                composites[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = this_cue
                masks[self.ep_no1:self.ep_no1_cue, :, h_slice, w_slice] = self.gaussian
            labels[self.ep_no1:self.ep_no1_cue] = 4

        # gabor
        if show_gabor:
            if self.cls_rot:
                rot_label = torch.randint(0, 4, (())).item()
                rot_noise = torch.randint(-self.rot_noise, self.rot_noise, (())).item()
                rotation = ((rot_label * 45) + rot_noise) % 180
                labels[-self.ep_stim:, 0] = gabor_loc
                labels[-self.ep_stim:, 1] = rot_label
            else:
                rotation = torch.randint(0, 180, (())).item()
                labels[-self.ep_stim:] = gabor_loc
            gabor = self.raw_gabors[rotation]
            gabor = gabor * amplitude
            i, j = divmod(gabor_loc, self.n_grid)
            h_slice, w_slice = self.make_slice(i, j)
            composites[-self.ep_stim:, :, h_slice, w_slice] = gabor
            masks[-self.ep_stim:, :, h_slice, w_slice] = self.gaussian
        else:
            labels[-self.ep_stim:] = 4

        # noise
        if self.decay_rate > 0.0:
            noise = self.ranger(self.n_range) * self.decay
            bias = min(self.ranger(self.b_range), 1.0 - amplitude)
            composites[:] += (noise[:, None, None, None] * torch.randn_like(composites) + bias)
        else:
            noise = self.ranger(self.n_range)
            bias = min(self.ranger(self.b_range), 1.0 - amplitude)
            composites[:] += (noise * torch.randn_like(composites) + bias)
        torch.clamp_(composites, 0.0, 1.0)
        labels = labels.long()
        masks = 2.0 * (masks - 0.5)
        
        return composites, labels, masks, components, hot_labels

