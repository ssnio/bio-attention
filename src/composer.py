# # built-in modules
import random
import os
from typing import Callable
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


def routine_01(composites: torch.Tensor, masks: torch.Tensor, noise: float = 0.0):
    # adding noise and clamping 
    composites += torch.rand(1) * noise * torch.rand_like(composites)
    composites = torch.clamp(composites, 0.0, 1.0)
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
            composites[:] += (x * rgb)
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
                 directory: str = r"./data/arrows",  # directory of the arrow images
                 ):
        
        super().__init__()
        self.directory = directory
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
        composites[:self.fixate] = fixpoint
        masks[:self.fixate] = fixpoint
        rgb = torch.rand(self.n_digits, 3, 1, 1)
        rgb /= rgb.max(dim=1, keepdims=True).values
        composites[self.fixate:] = (components * rgb).sum(0)
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
        b_rgb = torch.rand(3, 1, 1) * 0.5

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
