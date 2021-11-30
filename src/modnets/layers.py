"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import utils as utils
import numpy as np

DEFAULT_THRESHOLD = 5e-3



def _gumbel_sigmoid(x, gumbel_temp=1.0, gumbel_noise=False, thres=0, eps=1e-8, training=False):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    if not training:  # no Gumbel noise during inference
        return (x - thres >= 0).float()
    if gumbel_noise:
        with torch.no_grad():
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
        x = x + g1 - g2

    soft = torch.sigmoid(x / gumbel_temp)

    hard = ((soft >= 0.5).float() - soft).detach() + soft
    assert not torch.any(torch.isnan(hard))
    return hard


class GroupWiseConv2d(nn.Module): 
    """Modified conv with masks for weights on various structure sparsity pattern"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 mask_scale=1e-2,
                 threshold=None, mask='kernel_wise', soft=  True): # soft , quant = false ==> Binary. soft = True quan = false ==> KSM
        super(GroupWiseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale

        if threshold is None:
            threshold = DEFAULT_THRESHOLD

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.mask = mask
        self.soft = soft
        self.threshold = threshold
        self.n_groups = 8
        self.total_non_zeros= 0
        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask for various structured sparsity patterns.
        if self.mask == 'ch_wise':
            self.mask_real = self.weight.data.new(self.weight.size(0)).view(-1,1,1,1)
        elif self.mask == 'kernel_wise':
            self.mask_real = self.weight.data.new(self.weight.shape[:2]).view(self.weight.size(0),self.weight.size(1),1,1)
        elif self.mask == 'group_wise':
            self.mask_real = self.weight.data.new(self.out_channels, self.in_channels//self.n_groups, 1, 1)
        else:
            self.mask_real = self.weight.data.new(self.weight.size())
 
        # init
        self.mask_real.fill_(mask_scale)
   
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)
 
        # setting for gumbel
        self.temperature = 100
        self.Beta = 1200
        self.bin_mask = self.weight.data.new(self.mask_real.size())


    def forward(self, input):
        # if self.training:
        # bin_mask 
        one_hot = _gumbel_sigmoid((self.mask_real - DEFAULT_THRESHOLD) * self.Beta, gumbel_temp=self.temperature, training=self.training)
        self.total_non_zeros = one_hot.sum()
        if self.soft:
            invert_one_hot = 1 - one_hot
            a = self.mask_real.detach().clone() * invert_one_hot.detach()
            min_a = a.min()
            range_a = a.max() - a.min()
            a[a!=0] = (a[a!=0] - min_a)/range_a
            self.bin_mask = 0.5*a + one_hot
        else:
            self.bin_mask = one_hot         


        if self.mask == 'ch_wise':                                                                                                                                                                                                                                                          
            masked_weight = self.bin_mask[:,0].view(-1,1,1,1) * self.weight
        elif self.mask == 'kernel_wise':
            masked_weight = self.bin_mask.reshape(self.weight.size(0), self.weight.size(1),1,1) * self.weight           
        elif self.mask == 'group_wise':
            masked_weight = self.bin_mask.reshape(self.weight.size(0), self.weight.size(1)//self.n_groups, 1 ,1,1) * self.weight.reshape(self.weight.size(0), self.weight.size(1)//self.n_groups, self.n_groups, self.weight.size(2), self.weight.size(2)) 
            masked_weight = masked_weight.reshape(self.weight.size())
        else:
            masked_weight = self.bin_mask.reshape(self.weight.size()) * self.weight 

 
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)












                               