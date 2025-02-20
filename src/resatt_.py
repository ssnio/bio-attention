import torch
import torchvision
from torchvision.transforms import Normalize

def makenorm(kind: str, n_channels: int, trs: bool = False, aff: bool = True):
    if kind == 'batch':
        return torch.nn.BatchNorm2d(n_channels, affine=aff, track_running_stats=trs)
    elif kind == 'layer':
        return torch.nn.GroupNorm(1, n_channels, affine=aff)
    elif kind == 'instance':
        return torch.nn.InstanceNorm2d(n_channels, affine=aff, track_running_stats=trs)
    elif kind is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid normalization layer: {kind}!")
    

def get_dims(in_dims, out_channels):
    in_channels, in_h, in_w = in_dims
    out_h = in_dims[1] if in_channels == out_channels else in_dims[1] // 2
    out_w = in_dims[2] if in_channels == out_channels else in_dims[2] // 2
    return out_channels, out_h, out_w


class Block(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm: str,
                 fun: torch.nn.Module,
                 fsingle: bool = False,
                 bsingle: bool = False,
                 residual: bool = False,
                 ):
        super().__init__()
        self.fun = fun
        self.residual = residual
        ds = in_channels != out_channels
        bias = True if norm == 'layer' else False
        if ds:
            self.downsample = torch.nn.Sequential(
               torch.nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=bias),
                makenorm(norm, out_channels)
            )
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                makenorm(norm, 2 * out_channels)
            )
        else:
            self.downsample = torch.nn.Identity()
            self.upsample = torch.nn.Identity()
        if residual:
            self.deconv_res = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(2 * out_channels, in_channels, 2, 2, 0, bias=bias),
                makenorm(norm, in_channels)
            ) if ds else torch.nn.Sequential(
                torch.nn.ConvTranspose2d(2 * out_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels)
            )
        if fsingle:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, 1 if not ds else 2, 1, bias=bias),
                makenorm(norm, out_channels)
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
                makenorm(norm, out_channels),
                fun,
                torch.nn.Conv2d(out_channels, out_channels, 3, 1 if not ds else 2, 1, bias=bias),
                makenorm(norm, out_channels)
            )
        if bsingle:
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(2 * out_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels)
            )
        else:
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(2 * out_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels),
                fun,
                torch.nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels)
            )

    def forward(self, x: torch.Tensor):
        return self.fforward(x)

    def fforward(self, x: torch.Tensor):
        identity = self.downsample(x)
        x = self.conv(x)
        x += identity
        x = self.fun(x)
        return x

    def bforward(self, x: torch.Tensor):
        identity = self.deconv_res(x) if self.residual else None
        x = self.upsample(x)
        x = self.deconv(x)
        x = (x + identity) if self.residual else x
        x = torch.tanh(x)
        return x


class AttentionModel(torch.nn.Module):
    def __init__(self,
                 channels = [32, 64, 128, 256],
                 in_dims = (3, 256, 256),
                 fun = torch.nn.GELU(),
                 task_fun = torch.nn.Tanh(),
                 norm = 'layer',
                 out_dim = 20,
                 n_classes = 10,
                 mid_dim = 512,
                 n_tasks = 3,
                 softness = 0.5,
                 residual = False,
                 skip_maxpool = False,
                 first_k = 7,
                 ):
        
        super().__init__()
        self.channels = channels
        self.in_dims = in_dims
        self.fun = fun
        self.task_fun = task_fun
        self.norm = norm
        bias = True if norm == 'layer' else False
        self.out_dim = out_dim
        self.n_classes = n_classes
        self.mid_dim = mid_dim
        self.n_tasks = n_tasks
        self.softness = softness
        self.residual = residual
        self.skip_maxpool = skip_maxpool
        self.first_k, self.first_p = first_k, (first_k - 1)//2
        self.map_dims = [(1, self.in_dims[1], self.in_dims[2])]
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.channels[0], self.first_k, 2, self.first_p, bias=bias),
            makenorm(self.norm, self.channels[0]),
            self.fun,
            torch.nn.Identity() if skip_maxpool else torch.nn.MaxPool2d(3, 2, 1)
        )
        h, w = (hw//(2 if skip_maxpool else 4) for hw in self.in_dims[1:])
        self.map_dims.append((self.channels[0], h, w))
        self.blocks = torch.nn.ModuleList()
        for in_c, out_c in zip(self.channels[:-1], self.channels[1:]):
            self.blocks.append(Block(in_c, out_c, self.norm, self.fun, residual=self.residual))
            self.map_dims.append(get_dims(self.map_dims[-1], out_c))
        self.last_deconv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2 if skip_maxpool else 4),
            makenorm(self.norm, 2 * self.channels[0]),
            torch.nn.ConvTranspose2d(2 * self.channels[0], 1, 3, 1, 1, bias=bias),
            torch.nn.Tanh()
        )
        self.fmid = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(self.channels[-1], self.mid_dim),
            self.fun,
        )
        self.fbottleneck = torch.nn.Linear(self.mid_dim, self.out_dim)
        self.bbottleneck = torch.nn.Linear((self.out_dim + self.n_tasks) if self.n_tasks > 1 else self.out_dim, self.mid_dim)
        self.bmid = torch.nn.Sequential(
            torch.nn.Linear(self.mid_dim, self.channels[-1]),
            torch.nn.Unflatten(1, (self.channels[-1], 1, 1)),
            self.fun,
        )
        if self.n_tasks > 1:
            self.task_w = torch.torch.nn.Embedding(self.n_tasks, 2 * self.channels[-1])
            torch.nn.init.xavier_normal_(self.task_w.weight)
            self.task_b = torch.torch.nn.Embedding(self.n_tasks, 2 * self.channels[-1])
            torch.nn.init.zeros_(self.task_b.weight)

        self.masks = {}
        self.bmv = {}
        self.bmv_stuff = []
        self.bmv_i = {}

    # def make_task_iter_batch(self, device):
    #     for a in self.modules():
    #         if isinstance(a, torch.nn.BatchNorm2d):
    #             if a.track_running_stats == True:
    #                 for t in range(self.n_tasks):
    #                     self.bmv[f"task_{t}{0}"] = (torch.zeros(a.num_features).to(device), torch.ones(a.num_features).to(device))

    def set_task_iter_batch(self, device, t, i):
        j = 0
        for a in self.modules():
            if isinstance(a, torch.nn.BatchNorm2d):
                if a.running_mean is not None and a.running_var is not None:
                    # print("jere")
                    j += 1
                    if f"{t}_{i}_{j}" not in self.bmv_stuff:
                        # print(f"{t}_{i}_{j}")
                        self.bmv[f"{t}_{i}_{j}"] = (torch.zeros(a.num_features).to(device), torch.ones(a.num_features).to(device))
                        self.bmv_stuff.append(f"{t}_{i}_{j}")
                    a.running_mean, a.running_var = self.bmv[f"{t}_{i}_{j}"]
        self.bmv_i[f"{t}"] = max(self.bmv_i.get(f"{t}", 0), i)

    def prepare_task(self, t: int, batch_size: int, device):
        t = torch.tensor([t]).to(device).expand(batch_size).contiguous()
        th = torch.nn.functional.one_hot(t, self.n_tasks).contiguous().float()
        return t, th

    def initiate_forward(self, batch_size: int):
        device = next(self.parameters()).device
        for i, m in enumerate(self.map_dims):
            self.masks[f"mask_{i}"] = torch.zeros(batch_size, *m).to(device)

    def pre_allocation(self, n_iter: int, batch_size: int, device):
        masks_ = torch.empty(n_iter, batch_size, *self.map_dims[0]).to(device)
        act_ = []  # forward activation
        for m in self.map_dims[1:]:
            act_.append(torch.empty(n_iter, batch_size, *m).to(device))
        labels_ = torch.empty(n_iter, batch_size, self.n_classes).to(device)
        return masks_, act_, labels_

    def forward(self, x: torch.Tensor, t: int = None, y: torch.Tensor = None):
        # pre-processing
        device = next(self.parameters()).device
        batch_size, n_iter = x.shape[:2]
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        if t is not None and self.n_tasks > 1:
            t, th = self.prepare_task(t, batch_size, device)
        else:
            t, th = None, None
        if y is not None:
            y = y.permute(1, 0, 2).contiguous()

        # initialization
        self.initiate_forward(batch_size)

        # pre-allocation
        masks_, act_, labels_ = self.pre_allocation(n_iter, batch_size, device)

        for r in range(n_iter):  # Recurrent
            self.set_task_iter_batch(device, t[0].item(), r) if t is not None else None
            h = self.normalize(x[r])
            
            h = h * (1.0 + self.softness * self.masks[f"mask_{0}"])
            h = self.first_conv(h)
            act_[0][r] = h
            
            # convolutional layers
            for i, b_ in enumerate(self.blocks):
                h = h * (1.0 + self.softness * self.masks[f"mask_{i+1}"])
                h = b_.fforward(h)
                act_[i+1][r] = h

            # bottleneck
            h = self.fmid(h)
            h = self.fbottleneck(h)
            labels_[r] = h[:, :self.n_classes]
            h = h if y is None else torch.cat([y[r], h[:, self.n_classes:]], dim=1)
            h = h if t is None else torch.cat([h, th], 1) if self.n_tasks > 1 else h
            h = self.bbottleneck(h)
            h = self.bmid(h)
            h = h.repeat(1, 1, self.map_dims[-1][1], self.map_dims[-1][2])

            # backward # deconvolutional
            h = torch.cat([h, act_[-1][r]], 1)
            if self.n_tasks > 1:
                a = self.task_w(t).unsqueeze(-1).unsqueeze(-1)
                b = self.task_b(t).unsqueeze(-1).unsqueeze(-1)
                h = self.task_fun(a * h + b)
            
            for i, b_ in enumerate(self.blocks[::-1]):
                h = torch.cat([h, act_[-i-1][r]], 1) if i > 0 else h
                self.masks[f"mask_{len(self.blocks) - i}"] = h = b_.bforward(h)
            h = torch.cat([h, act_[0][r]], 1)
            masks_[r] = self.masks["mask_0"] = self.last_deconv(h)
            
        # post-processing
        labels_ = labels_.permute(1, 2, 0).contiguous()
        masks_ = masks_.swapaxes(0, 1).contiguous()
        for i in range(len(act_)):
            act_[i] = act_[i].swapaxes(0, 1).contiguous()
        
        return masks_, labels_, act_

    def for_forward(self, x: torch.Tensor, t: int = None):
        # pre-processing
        device = next(self.parameters()).device
        batch_size = x.size(0)

        # pre-allocation
        act_ = []  # forward activation
        for m in self.map_dims[1:]:
            act_.append(torch.empty(batch_size, *m).to(device))

        self.set_task_iter_batch(device, t, self.bmv_i[f"{t}"]) if t is not None else None
        h = self.normalize(x)
        
        h = h * (1.0 + self.softness * self.masks[f"mask_{0}"])
        h = self.first_conv(h)
        act_[0] = h
        
        # convolutional layers
        for i, b_ in enumerate(self.blocks):
            h = h * (1.0 + self.softness * self.masks[f"mask_{i+1}"])
            h = b_.fforward(h)
            act_[i+1] = h
        
        # forward connection linear layer
        h = self.fmid(h)

        # bottleneck
        h = self.fbottleneck(h)
        labels_ = h[:, :self.n_classes]
        
        return labels_, act_
