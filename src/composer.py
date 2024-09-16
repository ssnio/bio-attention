# # built-in modules
import random
import os
from typing import Callable, Union
import math
# # Torch modules
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.nn.functional import one_hot, conv2d
# # other modules
from PIL import Image as PILImage


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
    composites += torch.rand(1) * noise * torch.rand_like(composites)
    composites = torch.clamp(composites, 0.0, 1.0)
    masks = torch.clamp(masks, 0.0, 1.0)
    masks = 2.0 * (masks - 0.5)
    return composites, masks


class FixPoints:
    """Creates a fix point for a given object (digit) with size k x k
    such that the fix point is contiguous with the object."""
    def __init__(self, k: int = 3):
        self.k = k
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
        i, j = self.get_rand_fix_point(x)
        fix_points[:, i-self.p:i+self.p, j-self.p:j+self.p] = 1.0
        return fix_points


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
                 n_iter: tuple,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 directory: str = r"./data/",  # directory of the arrow images
                 ):
        
        super().__init__()
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
                x, _ = self.rand_sample(y, exclude=True)
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
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.stride = stride
        self.blank = blank
        self.static = static
        self.background = background
        self.noise = noise
        self.pad = 34
        self.c, h, w = self.dataset[0][0].shape
        self.h, self.w = self.pad + h + self.pad, self.pad + w + self.pad
        self.transform = transforms.Compose([
            transforms.Pad(self.pad),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.3, 0.3), scale=(1.3, 1.5))])
        
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
        
        digit_color = torch.rand(3, 1, 1)
        obstacle_color = 1 - digit_color
        digit_color /= digit_color.max()
        obstacle_color /= obstacle_color.max()
        
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
        foreground_color /= (foreground_color.max() + 1e-6) * 2
        background_color /= (background_color.max() + 1e-6) * 2
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
                 n_iter: tuple,  # number of iterations
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
                 n_iter: tuple,  # number of iterations
                 noise: float = 0.25,  # noise scale
                 ):
        
        super().__init__()
        self.dataset = mnist_dataset
        self.n_iter = n_iter
        self.noise = noise
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
            cls = y
        i = self.class_ids[cls][torch.randint(0, len(self.class_ids[cls]), (1, )).item()]
        return self.dataset.__getitem__(i)

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
        rgbs = torch.rand(2, 3, 1, 1)
        labels[:] = y
        hot_labels = 0
        b_rgb = 0.5 * torch.rand(3, 1, 1)

        t = random.choice(self.index_pos)
        pos_ij = self.digit_pos[t]
        composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = (x * rgbs[0]) + (1.0 - x) * b_rgb
        masks[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] += x
        
        x, _ = self.rand_sample(y, exclude=True)
        x = self.transform(x)
        for i in self.index_pos:
            if i != t:
                pos_ij = self.digit_pos[i]
                composites[:, :, pos_ij[0]:pos_ij[0]+32, pos_ij[1]:pos_ij[1]+32] = x * rgbs[1] + (1.0 - x) * b_rgb
        
        # adding noise and clamping 
        composites, masks = routine_01(composites, masks, self.noise)

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
        self.hair_dir = hair_dir if hair_dir is not None else r"./data/celeba/"
        self.in_dims = in_dims
        _, self.h, self.w = self.in_dims
        self.padding = padding
        self.noise = noise if kind == "train" else 0.0
        self.kind = kind  # train, valid, test
        self.which = which  # 0: all, 1: fblonde, 2: fbrunette, 3: mblonde, 4: mbrunette
        self.which_names = ["all", "fblonde", "fbrunette", "mblonde", "mbrunette"]
        self.len_which = len(self.which_names)
        assert self.which in range(self.len_which), f"which must be between 0 and {self.len_which-1} but got {self.which}!"
        self.at_list = ['Male', 'Black_Hair', 'Blond_Hair']
        self.gender_i, self.brunette_i, self.blonde_i =(self.dataset.attr_names.index(x) for x in self.at_list)
        self.hair_ids = self.get_hair()
        self.transform = transforms.Compose([
            transforms.Resize((self.h - 2*self.padding, self.w - 2*self.padding), antialias=True),
            transforms.RandomHorizontalFlip(),
            self.nosie_pad if self.kind == "train" else transforms.Pad(self.padding),
            ])

    def get_hair(self):
        if self.hair_dir and os.path.exists(os.path.join(self.hair_dir, f"{self.kind}_hair_ids.pt")):
            print(f'Loading {self.kind}_hair_ids.pt from file!')
            return torch.load(os.path.join(self.hair_dir, f"{self.kind}_hair_ids.pt"))
        else:
            print(f'Creating {self.kind}_hair_ids.pt file!')
            hair_ids = [[], [], [], [], []]  # [all, fblonde, fbrunette, mblonde, mbrunette]
            for i, (_, y) in enumerate(self.dataset):
                if y[self.blonde_i] == 1 or y[self.brunette_i] == 1:
                    hair_ids[0].append(i)
                    if self.kind == "train" and y[self.gender_i] == 1:  # since the dataset is not balanced
                        hair_ids[0].append(i)
                    g, b = y[self.gender_i], y[self.brunette_i]
                    hair_ids[1+b+2*g].append(i)
        if self.kind == "train":
            random.shuffle(hair_ids[0])  # shuffle the training set only # # # # 
        for i, n in enumerate(self.which_names):
            print(f"{n}: {len(hair_ids[i])}")
        torch.save(hair_ids, os.path.join(self.hair_dir, f"{self.kind}_hair_ids.pt"))
        print(f'{self.kind}_hair_ids.pt saved to file!')
        return hair_ids

    def nosie_pad(self, x: torch.Tensor):
        y = torch.rand(x.size(0), self.h, self.w)
        h, w = x.shape[-2:]
        pad_h = torch.randint(0, self.h - h, (1,)).item() if h < self.h else 0
        pad_w = torch.randint(0, self.w - w, (1,)).item() if w < self.w else 0
        y[:, pad_h:pad_h+h, pad_w:pad_w+w] = x
        return y

    def __len__(self):
        return len(self.hair_ids[self.which])

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.hair_ids[self.which][idx]]
        x = x[:, 20:-20, :]
        composites = torch.zeros(self.n_iter, 3, self.h, self.w)
        composites[:] = self.transform(x)
        labels = torch.zeros(self.n_iter).long()
        labels[:] = y[self.gender_i]
        composites += self.noise * torch.rand_like(composites)
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
                 ):
        super().__init__()
        in_dims = in_dims if len(in_dims) == 2 else in_dims[1:]
        from src.pycocotools.coco import COCO
        self.h, self.w = in_dims
        self.kind = kind
        self.tokens = tokens
        self.min_area = min_area
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

            self._get_class_weights()
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
    

class ConceptualGrouping_COCO(Dataset):
    def __init__(self,
                 coco_dataset: COCOAnimals,
                 fix_attend: tuple,
                 noise: float = 0.25,):
        
        super().__init__()
        self.kind = "train"
        assert len(fix_attend) == 2
        self.k = 3
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
        if torch.rand(1).item() > 0.5 or self.kind == "not_train":
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
        composites[:self.fixate] = p
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
                 n_iter: tuple,
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
                 n_iter: tuple,
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
                 n_iter: tuple,
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
                 n_iter: tuple,
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
                composites[:] = background[:, :, :self.w] * (1.0 - m) + x
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
