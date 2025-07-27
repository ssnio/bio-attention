import torch
import torchvision
from torchvision.transforms import Normalize

# def start_stop_forward_grads(model: torch.nn.Module, state: bool):
#     # START GRAD for forward layers 
#     for a in model.modules():
#         if isinstance(a, Block):
#             for m in a.modules():
#                 m.downsample.requires_grad_(state)
#                 m.conv.requires_grad_(state)

def makenorm(kind: str, n_channels: int, trs: bool = True, aff: bool = True):
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
    
def makefun(kind: str):
    if kind == 'tanh':
        return torch.nn.Tanh()
    elif kind == 'relu':
        return torch.nn.ReLU()
    elif kind == 'gelu':
        return torch.nn.GELU()
    else:
        raise ValueError(f"Invalid nonlinear function: {kind}!")


def get_dims(in_dims, out_channels):
    in_channels, in_h, in_w = in_dims
    out_h = in_dims[1] if in_channels == out_channels else in_dims[1] // 2
    out_w = in_dims[2] if in_channels == out_channels else in_dims[2] // 2
    return out_channels, out_h, out_w


class MonoSeqRNN(torch.nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hid_dim: int, 
                 bias: bool = False,
                 dropout: float = 0.0,
                 fun: torch.nn.Module = torch.nn.ReLU(),):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.bias = bias
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.nonlinearity = fun
        
        self.weight_ih_l0 = torch.nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(hid_dim, in_dim)))
        self.weight_hh_l0 = torch.nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(hid_dim, hid_dim)))
        if bias:
            self.bias_ih_l0 = torch.nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, hid_dim)))
            self.bias_hh_l0 = torch.nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(1, hid_dim)))

    def __call__(self, x: torch.Tensor, h: torch.Tensor):
        return self.forward(x, h)

    def __repr__(self) -> str:
        try:
            nonlinearity = self.nonlinearity.__name__
        except AttributeError:
            nonlinearity = "unknown"
        return f"MonoSeqRNN(in_dim={self.in_dim}, hid_dim={self.hid_dim}, nonlinearity={nonlinearity}, bias={self.bias})"

    def __str__(self) -> str:
        try:
            nonlinearity = self.nonlinearity.__name__
        except AttributeError:
            nonlinearity = "unknown"
        return f"MonoSeqRNN(in_dim={self.in_dim}, hid_dim={self.hid_dim}, nonlinearity={nonlinearity}, bias={self.bias})"

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        if self.bias:
            z = x @ self.weight_ih_l0.T + self.bias_ih_l0 + h @ self.weight_hh_l0.T + self.bias_hh_l0
            z = self.dropout(z) if self.dropout is not None else z
            return self.nonlinearity(z)
        else:
            z = x @ self.weight_ih_l0.T + h @ self.weight_hh_l0.T
            z = self.dropout(z) if self.dropout is not None else z
            return self.nonlinearity(z)


class Block(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm: str,
                 fun: torch.nn.Module,
                 fsingle: bool = False,
                 bsingle: bool = False,
                 residual: bool = False,
                 concat: bool = True,
                 resnet: bool = True,
                 ):
        super().__init__()
        self.fun = fun
        self.residual = residual
        ds = in_channels != out_channels
        bias = True if norm == 'layer' else False
        self.pre_conv_norm = makenorm(norm, in_channels)
        self.post_conv_norm = makenorm(norm, out_channels)
        self.post_deconv_norm = makenorm(norm, in_channels)
        self.ncu = 2 if concat else 1
        self.resnet = resnet
        if ds:
            self.downsample = torch.nn.Sequential(
               torch.nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=bias),
                makenorm(norm, out_channels)
            ) if self.resnet else None
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                makenorm(norm, self.ncu * out_channels)
            )
        else:
            self.downsample = torch.nn.Identity() if self.resnet else None
            self.upsample = torch.nn.Identity()
        if residual:
            self.deconv_res = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(self.ncu * out_channels, in_channels, 2, 2, 0, bias=bias),
                makenorm(norm, in_channels)
            ) if ds else torch.nn.Sequential(
                torch.nn.ConvTranspose2d(self.ncu * out_channels, in_channels, 3, 1, 1, bias=bias),
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
                torch.nn.ConvTranspose2d(self.ncu * out_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels)
            )
        else:
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(self.ncu * out_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels),
                fun,
                torch.nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=bias),
                makenorm(norm, in_channels)
            )

    def forward(self, x: torch.Tensor):
        return self.fforward(x)

    def fforward(self, x: torch.Tensor):
        identity = self.downsample(x) if self.resnet else None
        x = self.conv(x)
        x = (x + identity) if self.resnet else x
        x = self.post_conv_norm(x) if self.resnet else x
        x = self.fun(x)
        return x

    def bforward(self, x: torch.Tensor):
        identity = self.deconv_res(x) if self.residual else None
        x = self.upsample(x)
        x = self.deconv(x)
        x = (x + identity) if self.residual else x
        x = self.post_deconv_norm(x) if self.residual else x
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
                 bnpti: bool = False,
                 reinit: bool = False,
                 recurrent: bool = False,
                 concat: bool = True,
                 resnet: bool = True,
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
        self.bnpti = bnpti
        self.reinit = reinit
        self.recurrent = recurrent
        self.concat = concat
        self.resnet = resnet
        self.ncu = 2 if self.concat else 1
        self.first_k, self.first_p = first_k, (first_k - 1)//2
        self.map_dims = [(1, self.in_dims[1], self.in_dims[2])]
        if self.in_dims[0] == 3:
            self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif self.in_dims[0] == 1:
            self.normalize = Normalize([0.5], [0.25])
        else:
            raise ValueError(f"Invalid input dimensions: {self.in_dims[0]} channels! Only 1 or 3 channels are supported.")
        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_dims[0], self.channels[0], self.first_k, 2, self.first_p, bias=bias),
            makenorm(self.norm, self.channels[0]),
            self.fun,
            torch.nn.Identity() if skip_maxpool else torch.nn.MaxPool2d(3, 2, 1)
        )
        h, w = (hw//(2 if skip_maxpool else 4) for hw in self.in_dims[1:])
        self.map_dims.append((self.channels[0], h, w))
        self.blocks = torch.nn.ModuleList()
        for in_c, out_c in zip(self.channels[:-1], self.channels[1:]):
            self.blocks.append(Block(in_c, out_c, self.norm, self.fun, residual=self.residual, concat=self.concat, resnet=self.resnet))
            self.map_dims.append(get_dims(self.map_dims[-1], out_c))
        self.last_deconv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2 if skip_maxpool else 4),
            makenorm(self.norm, self.ncu * self.channels[0]),
            torch.nn.ConvTranspose2d(self.ncu * self.channels[0], 1, 3, 1, 1, bias=bias),
            torch.nn.Tanh()
        )
        self.fmid = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        if self.mid_dim < 0:
            self.frnn = torch.nn.Identity()
        elif self.recurrent:
            self.frnn = MonoSeqRNN(self.channels[-1], self.mid_dim, True, 0.0, self.fun)
        else:
            self.frnn = torch.nn.Sequential(torch.nn.Linear(self.channels[-1], self.mid_dim), self.fun)

        self.fbottleneck = torch.nn.Linear(self.channels[-1] if self.mid_dim < 0 else self.mid_dim, self.out_dim)
        self.bbottleneck = torch.nn.Linear((self.out_dim + self.n_tasks) if self.n_tasks > 1 else self.out_dim, self.channels[-1] if self.mid_dim < 0 else self.mid_dim)

        if self.mid_dim < 0:
            self.brnn = torch.nn.Identity()
        elif self.recurrent:
            self.brnn = MonoSeqRNN(self.mid_dim, self.channels[-1], True, 0.0, self.fun)
        else:
            self.brnn = torch.nn.Sequential(torch.nn.Linear(self.mid_dim, self.channels[-1]), self.fun)

        self.bmid = torch.nn.Unflatten(1, (self.channels[-1], 1, 1))

        if self.n_tasks > 1:
            self.task_w = torch.torch.nn.Embedding(self.n_tasks, self.ncu * self.channels[-1])
            torch.nn.init.xavier_normal_(self.task_w.weight)
            self.task_b = torch.torch.nn.Embedding(self.n_tasks, self.ncu * self.channels[-1])
            torch.nn.init.zeros_(self.task_b.weight)

        # if self.reinit:
        #     for m in self.modules():
        #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        #             torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #             # if isinstance(self.fun, torch.nn.ReLU):
        #             #     torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #             # else:
        #             #     torch.nn.init.xavier_normal_(m.weight, gain=0.1)
        #         elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        #             torch.nn.init.constant_(m.weight, 1)
        #             torch.nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            # for m in self.modules():
            #     if isinstance(m, Block) and m.conv[-1].weight is not None:
            #         torch.nn.init.constant_(m.conv[-1].weight, 0)  # type: ignore[arg-type]
            #     elif isinstance(m, Block) and m.deconv[-1].weight is not None:
            #         torch.nn.init.constant_(m.deconv[-1].weight, 0)  # type: ignore[arg-type]

        self.masks = {}
        self.hstates = {}
        self.bmv = torch.nn.ModuleDict()
        self.bmv_stuff = []

    # def toggle_forward(self, state: bool):
    #     self.first_conv.requires_grad_(state)
    #     self.fmid.requires_grad_(state)
    #     self.fbottleneck.requires_grad_(state)

    #     for a in self.modules():
    #         if isinstance(a, Block):
    #             a.downsample.requires_grad_(state)
    #             a.conv.requires_grad_(state)

    # def set_task_iter_batch(self, device, t, i):
    #     j = 0
    #     for a in self.modules():
    #         if isinstance(a, torch.nn.BatchNorm2d):
    #             if a.running_mean is not None and a.running_var is not None:
    #                 if f"{t}_{i}_{j}" not in self.bmv_stuff:
    #                     self.bmv.register_buffer(f"{t}_{i}_{j}_rm", torch.zeros(a.num_features).to(device))
    #                     self.bmv.register_buffer(f"{t}_{i}_{j}_rv", torch.ones(a.num_features).to(device))
    #                     self.bmv_stuff.append(f"{t}_{i}_{j}")
    #                 a.running_mean = self.bmv.get_buffer(f"{t}_{i}_{j}_rm")
    #                 a.running_var = self.bmv.get_buffer(f"{t}_{i}_{j}_rv")
    #                 j += 1

    # def get_task_iter_batch(self, device, t, i):
    #     j = 0
    #     for a in self.modules():
    #         if isinstance(a, torch.nn.BatchNorm2d):
    #             if a.training:
    #                 if f"{t}_{i}_{j}" in self.bmv_stuff:
    #                     self.bmv.__setattr__(f"{t}_{i}_{j}_rm", a.running_mean)
    #                     self.bmv.__setattr__(f"{t}_{i}_{j}_rv", a.running_var)
    #                     j += 1

    def prepare_task(self, t: int, batch_size: int, device):
        t = torch.tensor([t]).to(device).expand(batch_size).contiguous()
        th = torch.nn.functional.one_hot(t, self.n_tasks).contiguous().float()
        return t, th

    def initiate_forward(self, batch_size: int):
        device = next(self.parameters()).device
        for i, m in enumerate(self.map_dims):
            self.masks[f"mask_{i}"] = torch.zeros(batch_size, *m).to(device)
        if self.recurrent:
            self.hstates["fh"] = torch.zeros(batch_size, self.mid_dim).to(device)
            self.hstates["bh"] = torch.zeros(batch_size, self.channels[-1]).to(device)

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
            # t, th = None, None
            th = None
        if y is not None:
            y = y.permute(1, 0, 2).contiguous()

        # initialization
        self.initiate_forward(batch_size)

        # pre-allocation
        masks_, act_, labels_ = self.pre_allocation(n_iter, batch_size, device)

        for r in range(n_iter):  # Recurrent
            h = self.normalize(x[r])
            
            h = h * (1.0 + self.softness * self.masks[f"mask_{0}"])
            h = self.first_conv(h)
            act_[0][r] = h
            
            # convolutional layers
            for i, b_ in enumerate(self.blocks):
                h = h * (1.0 + self.softness * self.masks[f"mask_{i+1}"])
                # h = b_.pre_conv_norm(h) if r > 0 else h
                h = b_.pre_conv_norm(h)# if r > 0 else h
                h = b_.fforward(h)
                act_[i+1][r] = h

            # bottleneck
            h = self.fmid(h)
            h = self.frnn(h, self.hstates["fh"]) if self.recurrent else self.frnn(h)
            h = self.fbottleneck(h)
            labels_[r] = h[:, :self.n_classes]
            h = h if y is None else torch.cat([y[r], h[:, self.n_classes:]], dim=1)
            h = h if t is None else torch.cat([h, th], 1) if self.n_tasks > 1 else h
            h = self.bbottleneck(h)
            h = self.brnn(h, self.hstates["bh"]) if self.recurrent else self.brnn(h)
            h = self.bmid(h)
            h = h.repeat(1, 1, self.map_dims[-1][1], self.map_dims[-1][2])

            # backward # deconvolutional
            h = torch.cat([h, act_[-1][r]], 1) if self.concat else (h + act_[-1][r])
            if self.n_tasks > 1:
                a = self.task_w(t).unsqueeze(-1).unsqueeze(-1)
                b = self.task_b(t).unsqueeze(-1).unsqueeze(-1)
                h = self.task_fun(a * h + b)
            
            for i, b_ in enumerate(self.blocks[::-1]):
                h = (torch.cat([h, act_[-i-1][r]], 1) if self.concat else (h + act_[-i-1][r])) if i > 0 else h
                self.masks[f"mask_{len(self.blocks) - i}"] = h = b_.bforward(h)
            h = torch.cat([h, act_[0][r]], 1) if self.concat else (h + act_[0][r])
            masks_[r] = self.masks["mask_0"] = self.last_deconv(h)
            
        # post-processing
        labels_ = labels_.permute(1, 2, 0).contiguous()
        masks_ = masks_.swapaxes(0, 1).contiguous()
        for i in range(len(act_)):
            act_[i] = act_[i].swapaxes(0, 1).contiguous()
        
        return masks_, labels_, act_

    def for_forward(self, x: torch.Tensor, t: int = None, r: int = None):
        # pre-processing
        device = next(self.parameters()).device
        batch_size = x.size(0)

        # pre-allocation
        act_ = []  # forward activation
        for m in self.map_dims[1:]:
            act_.append(torch.empty(batch_size, *m).to(device))

        h = self.normalize(x)
        
        h = h * (1.0 + self.softness * self.masks[f"mask_{0}"])
        h = self.first_conv(h)
        act_[0] = h
        
        # convolutional layers
        for i, b_ in enumerate(self.blocks):
            h = h * (1.0 + self.softness * self.masks[f"mask_{i+1}"])
            h = b_.pre_conv_norm(h)
            h = b_.fforward(h)
            act_[i+1] = h
        
        # forward connection linear layer
        h = self.fmid(h)
        h = self.frnn(h, self.hstates["fh"]) if self.recurrent else self.frnn(h)

        # bottleneck
        h = self.fbottleneck(h)
        labels_ = h[:, :self.n_classes]
        
        return labels_, act_

    def one_forward(self, x: torch.Tensor, t: int = None, y: torch.Tensor = None):
        assert x.dim() == 4, "Input tensor should be 4D: B x C x H x W"
        
        # pre-processing
        device = next(self.parameters()).device
        batch_size = x.size(0)
        if t is not None and self.n_tasks > 1:
            t, th = self.prepare_task(t, batch_size, device)
        else:
            th = None

        # pre-allocation
        act_ = []  # forward activation
    
        h = self.normalize(x)
        h = h * (1.0 + self.softness * self.masks[f"mask_{0}"])
        h = self.first_conv(h)
        act_.append(h)
        
        # convolutional layers
        for i, b_ in enumerate(self.blocks):
            h = h * (1.0 + self.softness * self.masks[f"mask_{i+1}"])
            h = b_.pre_conv_norm(h)
            h = b_.fforward(h)
            act_.append(h)

        # bottleneck
        h = self.fmid(h)
        h = self.frnn(h, self.hstates["fh"]) if self.recurrent else self.frnn(h)
        h = self.fbottleneck(h)
        labels_ = h[:, :self.n_classes]
        h = h if y is None else torch.cat([y, h[:, self.n_classes:]], dim=1)
        h = h if t is None else torch.cat([h, th], 1) if self.n_tasks > 1 else h
        h = self.bbottleneck(h)
        h = self.brnn(h, self.hstates["bh"]) if self.recurrent else self.brnn(h)
        h = self.bmid(h)
        h = h.repeat(1, 1, self.map_dims[-1][1], self.map_dims[-1][2])

        # backward # deconvolutional
        h = torch.cat([h, act_[-1]], 1) if self.concat else (h + act_[-1])
        if self.n_tasks > 1:
            a = self.task_w(t).unsqueeze(-1).unsqueeze(-1)
            b = self.task_b(t).unsqueeze(-1).unsqueeze(-1)
            h = self.task_fun(a * h + b)
        
        for i, b_ in enumerate(self.blocks[::-1]):
            h = (torch.cat([h, act_[-i-1]], 1) if self.concat else (h + act_[-i-1])) if i > 0 else h
            self.masks[f"mask_{len(self.blocks) - i}"] = h = b_.bforward(h)
        h = torch.cat([h, act_[0]], 1) if self.concat else (h + act_[0])
        masks_ = self.masks["mask_0"] = self.last_deconv(h)

        return masks_, labels_, act_
