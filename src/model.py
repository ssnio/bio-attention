# # built-in modules
from typing import Callable
# # Torch modules
import torch
from torchvision.transforms.functional import normalize
# # internal imports
from .utils import obj_to_tuple, get_dims


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


class MonoSeqRNN(torch.nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hid_dim: int, 
                 bias: bool = False,
                 dropout: float = 0.0,
                 nonlinearity: Callable = torch.relu):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.bias = bias
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.nonlinearity = nonlinearity
        
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


class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 'same',
                 bias = True,
                 norm = None,
                 dropout = 0.0,
                 fun = torch.nn.ReLU(),
                 pool = None,
                 residual = False,
                 last = False,
                 affine = True,
                 ):
        super().__init__()
        self.residual = residual
        assert not residual or in_channels == out_channels, "Residual connection requires in_channels == out_channels"
        # assert padding == 'same' or not self.residual, "Padding 'same' is only supported for non-residual connections"
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode='reflect')
        self.norm = makenorm(norm, out_channels, False, affine)
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.fun = fun
        if pool is not None:
            if last:
                self.pool = torch.nn.AvgPool2d(pool)
            else:
                self.pool = torch.nn.MaxPool2d(pool)
        else:
            self.pool = None
    
    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = self.norm(h) if self.norm is not None else h
        h = self.dropout(h) if self.dropout is not None else h
        h = self.fun(h)
        h = h + x if self.residual else h
        h = self.pool(h) if self.pool is not None else h
        return h


class DeConvBlock(torch.nn.Module):
    def __init__(self,
                 upsample,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 'same',
                 bias = True,
                 norm = None,
                 dropout = 0.0,
                 fun = torch.nn.Tanh(),
                 affine = True,
                 ):
        super().__init__()

        padding = 1 if padding in ('same', 1) else 0
        self.upsample = torch.nn.Upsample(size=upsample)
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.norm = makenorm(norm, out_channels, False, affine)
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.fun = fun
    
    def forward(self, x: torch.Tensor):
        h = self.upsample(x)
        h = self.deconv(h)
        h = self.norm(h) if self.norm is not None else h
        h = self.dropout(h) if self.dropout is not None else h
        h = self.fun(h)
        return h


class AttentionModel(torch.nn.Module):
    def __init__(self,
                 in_dims: tuple,  # C x H x W
                 n_classes: int,  # number of classes
                 out_dim: int,  # output dimentions
                 normalize: bool,  # normalize input
                 softness: tuple,  # softness of attention
                 channels: tuple,  # convolutional channels
                 residuals: tuple,  # tuple of boolean for residual connections
                 kernels: tuple,  # kernel sizes for convolutional layers
                 strides: tuple,  # strides for convolutional layers
                 paddings: tuple,  # paddings 'same' or 'valid'
                 conv_bias: tuple,  # bias for convolutional layers
                 conv_norms: tuple,  # norm layer for convolutional layers ("batch" or "layer" or 'instance' or None)
                 conv_dropouts: tuple,  # dropout for convolutional layers
                 conv_funs: tuple,  # activation functions for convolutional layers
                 deconv_norms: tuple,  # norm layer for deconvolutional layers ("batch" or "layer" or 'instance' or None)
                 deconv_funs: tuple,  # activation functions for deconvolutional layers
                 pools: tuple,  # pooling layers
                 rnn_dims: tuple,  # dimensions of RNN layers
                 rnn_bias: tuple,  # bias for RNN layers
                 rnn_dropouts: tuple,  # dropout for RNN layers
                 rnn_funs: tuple,  # activation functions for RNN layers
                 n_tasks: int,  # number of tasks
                 norm_mean: float = None,  # mean for normalization
                 norm_std: float = None,  # std for normalization
                 task_layers: int = -1,  # number of layers to use task embedding for
                 task_weight: bool = True,  # whether to weight the deconvolutional layers with task embedding
                 task_bias: bool = True,  # whether to bias the deconvolutional layers with task embedding
                 task_funs: Callable = None,  # activation function for task embedding
                 rnn_to_fc: bool = False,  # whether to use RNN layers or MLP layers
                 trans_fun: Callable = torch.nn.Identity(),  # the activation function between convolutional and RNN layers
                 affine: bool = True,  # whether to use affine transformation in normalization layers
                 ):
        super().__init__()
        self.normalize = normalize
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.n_classes = n_classes
        assert self.out_dim >= self.n_classes, "Output dimensions should be greater than the number of classes!"
        assert len(in_dims) == 3, "Input dimensions should be 3D: C x H x W !"
        assert in_dims[0] == channels[0], "Input channels should match the first convolutional layer!"
        self.channels = channels
        self.n_convs = len(self.channels) - 1
        self.residuals = list(obj_to_tuple(residuals, self.n_convs))
        self.residuals[0] = False
        self.softness = obj_to_tuple(softness, self.n_convs)
        self.kernels = obj_to_tuple(kernels, self.n_convs)
        self.strides = obj_to_tuple(strides, self.n_convs)
        if isinstance(paddings, int):
            paddings = paddings
        elif paddings == 'same':
            paddings = list((1 if k == 3 else 2 if k == 5 else 3) for k in self.kernels)
        self.paddings = obj_to_tuple(paddings, self.n_convs)
        self.conv_bias = obj_to_tuple(conv_bias, self.n_convs)
        self.conv_norms = obj_to_tuple(conv_norms, self.n_convs)
        self.conv_dropouts = obj_to_tuple(conv_dropouts, self.n_convs)
        self.conv_funs = obj_to_tuple(conv_funs, self.n_convs)
        self.deconv_funs = obj_to_tuple(deconv_funs, self.n_convs)
        self.deconv_norms = obj_to_tuple(deconv_norms, self.n_convs)
        self.pools = obj_to_tuple(pools, self.n_convs)
        self.rnn_dims = rnn_dims
        self.n_rnns = len(self.rnn_dims) - 1
        self.rnn_bias = obj_to_tuple(rnn_bias, self.n_rnns + 1)
        self.rnn_dropouts = obj_to_tuple(rnn_dropouts, self.n_rnns + 1)
        self.rnn_funs = obj_to_tuple(rnn_funs, self.n_rnns + 1)
        self.n_tasks = n_tasks
        self.task_dim = self.n_tasks if self.n_tasks > 1 else 0
        self.norm_mean = [0.485, 0.456, 0.406] if norm_mean is None else norm_mean
        self.norm_std = [0.229, 0.224, 0.225] if norm_std is None else norm_std
        self.task_layers = list(range(self.n_convs)) if task_layers == -1 else list(range(task_layers))
        self.task_weight = task_weight if self.n_tasks > 1 else False
        self.task_bias = task_bias if self.n_tasks > 1 else False
        self.task_funs = task_funs if self.n_tasks > 1 else None
        self.conv_dims = [self.in_dims]
        self.frnn_to_fc = rnn_to_fc
        self.brnn_to_fc = rnn_to_fc
        self.bridge_norm = "layer"
        self.trans_fun = trans_fun
        self.affine = affine
        self.conv_blocks = torch.nn.ModuleList()
        self.frnn_blocks = torch.nn.ModuleList() if self.n_rnns > 0 else None
        self.brnn_blocks = torch.nn.ModuleList() if self.n_rnns > 0 else None
        self.deconv_blocks = torch.nn.ModuleList()
        self.embed_blocks_a = torch.nn.ModuleList() if self.task_weight else None
        self.embed_blocks_b = torch.nn.ModuleList() if self.task_bias else None
        for i in range(self.n_convs):
            self.conv_blocks.append(ConvBlock(self.channels[i], 
                                              self.channels[i+1], 
                                              self.kernels[i], 
                                              self.strides[i], 
                                              self.paddings[i], 
                                              self.conv_bias[i], 
                                              self.conv_norms[i], 
                                              self.conv_dropouts[i],
                                              self.conv_funs[i], 
                                              self.pools[i], 
                                              self.residuals[i],
                                              False if i < self.n_convs - 1 else True,
                                              affine=self.affine
                                              ))
            c, (h, w) = self.channels[i+1], get_dims(self.conv_dims[-1], self.conv_blocks[-1].conv)
            h, w = h // self.pools[i], w // self.pools[i]
            self.conv_dims.append((c, h, w))
        self.flat_dim = self.conv_dims[-1][0] * self.conv_dims[-1][1] * self.conv_dims[-1][2]
        self.conv_frnn = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.flat_dim, self.rnn_dims[0], bias=self.rnn_bias[0]),
            self.trans_fun,
        )
        
        for i in range(self.n_rnns):
            if self.frnn_to_fc:
                self.frnn_blocks.append(torch.nn.Sequential(
                    torch.nn.Linear(self.rnn_dims[i], self.rnn_dims[i+1], bias=self.rnn_bias[i]),
                    torch.nn.Dropout(self.rnn_dropouts[i]) if self.rnn_dropouts[i] > 0.0 else torch.nn.Identity(),
                    self.rnn_funs[i]))
            else:
                self.frnn_blocks.append(MonoSeqRNN(self.rnn_dims[i], 
                                                self.rnn_dims[i+1], 
                                                self.rnn_bias[i], 
                                                self.rnn_dropouts[i], 
                                                self.rnn_funs[i]))
        self.fc_out = torch.nn.Linear(self.rnn_dims[-1], out_dim)
        self.fc_in = torch.nn.Linear(self.task_dim + out_dim, self.rnn_dims[-1])
        for i in range(1, self.n_rnns + 1):
            if self.brnn_to_fc:
                self.brnn_blocks.append(torch.nn.Sequential(
                    torch.nn.Linear(self.rnn_dims[-i], self.rnn_dims[-i-1], bias=self.rnn_bias[-i]),
                    torch.nn.Dropout(self.rnn_dropouts[-i]) if self.rnn_dropouts[-i] > 0.0 else torch.nn.Identity(),
                    self.rnn_funs[-i]))
            else:
                self.brnn_blocks.append(MonoSeqRNN(self.rnn_dims[-i], 
                                                    self.rnn_dims[-i-1], 
                                                    self.rnn_bias[-i], 
                                                    self.rnn_dropouts[-i], 
                                                    self.rnn_funs[-i]))
        self.brnn_deconv = torch.nn.Sequential(
            torch.nn.Linear(self.rnn_dims[0], self.flat_dim, bias=self.rnn_bias[0]),
            self.trans_fun,
            torch.nn.Unflatten(1, self.conv_dims[-1]),
        )
        for i in range(1, self.n_convs + 1):
            if (i - 1) in self.task_layers:
                if self.task_weight:
                    self.embed_blocks_a.append(torch.torch.nn.Embedding(self.n_tasks, 2 * self.channels[-i]))
                    torch.nn.init.xavier_normal_(self.embed_blocks_a[-1].weight)
                    if self.task_bias:
                        self.embed_blocks_b.append(torch.torch.nn.Embedding(self.n_tasks, 2 * self.channels[-i]))
                        torch.nn.init.zeros_(self.embed_blocks_b[-1].weight)
            self.deconv_blocks.append(DeConvBlock(self.conv_dims[-i-1][-2:],
                                                  2 * self.channels[-i], 
                                                  self.channels[-i-1] if i < self.n_convs else 1,
                                                  3,
                                                  1,
                                                  'same',
                                                  self.conv_bias[-i],
                                                  self.deconv_norms[-i],
                                                  self.conv_dropouts[-i],
                                                  self.deconv_funs[-i] if i < self.n_convs else torch.nn.Tanh(), 
                                                  affine=self.affine
                                                  ))

        # pre-allocation
        self.masks = {}
        self.gates = {}
        self.hstates = {}

    def re_init(self, init_way, gain: float):
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim > 1:
                    init_way(p, gain=gain)
                else:
                    torch.nn.init.zeros_(p)

    def soft_attention(self, x: torch.Tensor, i: int):
        m = self.masks[f"mask_{i}"]
        return x * (1.0 + self.softness[i] * m)

    def initiate_forward(self, batch_size: int):
        device = next(self.parameters()).device
        for i in range(self.n_convs):
            self.masks[f"mask_{i}"] = torch.zeros(batch_size, *self.conv_dims[i]).to(device)
        for i in range(self.n_rnns):
            self.gates[f"gates_{i}"] = torch.zeros(batch_size, self.rnn_dims[i+1]).to(device)
        for i in range(self.n_rnns):
            self.hstates[f"f_state{i}"] = torch.zeros(batch_size, self.rnn_dims[i+1]).to(device)
            self.hstates[f"b_state{i}"] = torch.zeros(batch_size, self.rnn_dims[-i-2]).to(device)

    def prepare_task(self, t: int, batch_size: int, device):
        t = torch.tensor([t]).to(device).expand(batch_size).contiguous()
        th = torch.nn.functional.one_hot(t, self.n_tasks).contiguous().float()
        return t, th

    def pre_allocation(self, n_iter: int, batch_size: int, device):
        masks_ = torch.empty(n_iter, batch_size, 1, *self.in_dims[1:]).to(device)
        act_ = []  # forward activation
        for i in range(self.n_convs):
            act_.append(torch.empty(n_iter, batch_size, *self.conv_dims[i+1]).to(device))
        act_.append(torch.empty(n_iter, batch_size, self.rnn_dims[0]).to(device))
        for i in range(self.n_rnns):
            act_.append(torch.empty(n_iter, batch_size, self.rnn_dims[i+1]).to(device))
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
            if y.size(2) < self.n_classes:
                y = torch.nn.functional.pad(y, (0, self.n_classes - y.size(2)), mode='constant', value=0.0).to(device)
            y = y.permute(1, 0, 2).contiguous()

        # initialization
        self.initiate_forward(batch_size)

        # pre-allocation
        masks_, act_, labels_ = self.pre_allocation(n_iter, batch_size, device)

        for r in range(n_iter):  # Recurrent
            h = normalize(x[r], self.norm_mean, self.norm_std) if self.normalize else x[r]
            
            # convolutional layers
            for i in range(self.n_convs):
                h = self.soft_attention(h, i)
                h = self.conv_blocks[i](h)
                act_[i][r] = h
            
            # forward conv-rnn connection linear layer
            h = act_[self.n_convs][r] = self.conv_frnn(h)

            # forward recurrent layers
            for i in range(self.n_rnns):
                h = self.frnn_blocks[i](h) if self.frnn_to_fc else self.frnn_blocks[i](h, self.hstates[f"f_state{i}"])
                self.hstates[f"f_state{i}"] = h
                act_[self.n_convs + i + 1][r] = h

            # bottleneck (output labels, input prompts, tasks)
            h = self.fc_out(h)
            labels_[r] = h[:, :self.n_classes]
            h = h if y is None else torch.cat([y[r], h[:, self.n_classes:]], dim=1)
            h = h if t is None else torch.cat([h, th], 1) if self.n_tasks > 1 else h

            # backward recurrent layers
            h = self.fc_in(h)
            for i in range(self.n_rnns):
                h = self.brnn_blocks[i](h) if self.brnn_to_fc else self.brnn_blocks[i](h, self.hstates[f"b_state{i}"])
                self.hstates[f"b_state{i}"] = h
            
            # backward linear layer
            h = self.brnn_deconv(h)

            # deconvolutional layers
            for i in range(self.n_convs):
                f = act_[self.n_convs - i - 1][r]
                h = torch.cat([h, f], 1)
                if (t is not None) and (i in self.task_layers):
                    a = self.embed_blocks_a[i](t).unsqueeze(-1).unsqueeze(-1) if self.task_weight else 1.0
                    b = self.embed_blocks_b[i](t).unsqueeze(-1).unsqueeze(-1) if self.task_bias else 0.0
                    h = a * h + b if self.task_funs is None else self.task_funs(a * h + b)
                h = self.deconv_blocks[i](h)
                self.masks[f"mask_{self.n_convs - i - 1}"] = h
            masks_[r] = self.masks["mask_0"]

        # post-processing
        labels_ = labels_.permute(1, 2, 0).contiguous()
        masks_ = masks_.swapaxes(0, 1).contiguous()
        for i in range(len(act_)):
            act_[i] = act_[i].swapaxes(0, 1).contiguous()
        
        return masks_, labels_, act_

    def for_forward(self, x: torch.Tensor):
        # pre-processing
        device = next(self.parameters()).device
        batch_size = x.size(0)

        # pre-allocation
        act_ = []  # forward activation
        for i in range(self.n_convs):
            act_.append(torch.empty(batch_size, *self.conv_dims[i+1]).to(device))
        act_.append(torch.empty(batch_size, self.rnn_dims[0]).to(device))
        for i in range(self.n_rnns):
            act_.append(torch.empty(batch_size, self.rnn_dims[i+1]).to(device))

        h = normalize(x, self.norm_mean, self.norm_std) if self.normalize else x
        
        # convolutional layers
        for i in range(self.n_convs):
            h = self.soft_attention(h, i)
            h = self.conv_blocks[i](h)
            act_[i] = h
        
        # forward conv-rnn connection linear layer
        h = h.flatten(start_dim=1)
        h = self.conv_frnn(h)
        act_[self.n_convs] = h

        # forward recurrent layers
        for i in range(self.n_rnns):
            h = self.frnn_blocks[i](h) if self.frnn_to_fc else self.frnn_blocks[i](h, self.hstates[f"f_state{i}"])
            self.hstates[f"f_state{i}"] = h
            act_[self.n_convs + i + 1] = h

        # output
        h = self.fc_out(h)
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
            t, th = None, None
        if y is not None and y.size(1) < self.n_classes:
            y = torch.nn.functional.pad(y, (0, self.n_classes - y.size(1)), mode='constant', value=0.0).to(device)
        h = normalize(x, self.norm_mean, self.norm_std) if self.normalize else x

        # pre-allocation
        act_ = []  # forward activation
        
        # convolutional layers
        for i in range(self.n_convs):
            h = self.soft_attention(h, i)
            h = self.conv_blocks[i](h)
            act_.append(h)
        
        # forward conv-rnn connection linear layer
        h = h.flatten(start_dim=1)
        h = self.conv_frnn(h)
        act_.append(h)

        # forward recurrent layers
        for i in range(self.n_rnns):
            h = self.frnn_blocks[i](h) if self.frnn_to_fc else self.frnn_blocks[i](h, self.hstates[f"f_state{i}"])
            self.hstates[f"f_state{i}"] = h
            act_.append(h)

        # output and input prompt layer
        h = self.fc_out(h)
        labels_ = h[:, :self.n_classes]
        h = h if y is None else torch.cat([y, h[:, self.n_classes:]], dim=1)
        h = h if t is None else torch.cat([h, th], 1) if self.n_tasks > 1 else h

        # backward recurrent layers
        h = self.fc_in(h)
        for i in range(self.n_rnns):
            h = self.brnn_blocks[i](h) if self.brnn_to_fc else self.brnn_blocks[i](h, self.hstates[f"b_state{i}"])
            self.hstates[f"b_state{i}"] = h
        
        # backward linear layer
        h = self.brnn_deconv(h)
        h = h.view(-1, *self.conv_dims[-1])

        # deconvolutional layers
        for i in range(self.n_convs):
            f = act_[self.n_convs - i - 1]
            h = torch.cat([h, f], 1)
            if (t is not None) and (i in self.task_layers):
                a = self.embed_blocks_a[i](t).unsqueeze(-1).unsqueeze(-1) if self.task_weight else 1.0
                b = self.embed_blocks_b[i](t).unsqueeze(-1).unsqueeze(-1) if self.task_bias else 0.0
                h = a * h + b if self.task_funs is None else self.task_funs(a * h + b)
            h = self.deconv_blocks[i](h)
            self.masks[f"mask_{self.n_convs - i - 1}"] = h
        masks_ = self.masks["mask_0"]
        
        return masks_, labels_, act_


    def simp_forward(self, x: torch.Tensor):
        h = normalize(x, self.norm_mean, self.norm_std) if self.normalize else x
        # convolutional layers
        for i in range(self.n_convs):
            h = self.conv_blocks[i](h)
        
        # forward conv-rnn connection linear layer
        h = h.flatten(start_dim=1)
        h = self.conv_frnn(h)

        # forward recurrent layers
        for i in range(self.n_rnns):
            h = self.frnn_blocks[i](h) if self.frnn_to_fc else self.frnn_blocks[i](h, self.hstates[f"f_state{i}"])
            self.hstates[f"f_state{i}"] = h

        # output
        h = self.fc_out(h)
        labels_ = h[:, :self.n_classes]
        
        return labels_
