import torch
import torch.nn as nn
import numpy as np

DTYPE = torch.complex64
EPS = torch.finfo(DTYPE).eps * 10

def _roll(x, N):
    return torch.cat((x[-N:,...], x[:-N,...]), dim=0)

def _haar2(x, dtype=torch.complex64, device='cpu'):
    a0 = (1/2)*(x[0::2,0::2] + x[0::2,1::2] + x[1::2,0::2] + x[1::2,1::2])
    a1 = (1/2)*(x[0::2,0::2] + x[0::2,1::2] - x[1::2,0::2] - x[1::2,1::2])
    a2 = (1/2)*(x[0::2,0::2] - x[0::2,1::2] + x[1::2,0::2] - x[1::2,1::2])
    a3 = (1/2)*(x[0::2,0::2] - x[0::2,1::2] - x[1::2,0::2] + x[1::2,1::2])
    return [a0,a1,a2,a3]

def _ihaar2(a, dtype=torch.complex64, device='cpu'):
    Np = [a[0].shape[0] * 2, a[0].shape[1] * 2]
    x = torch.zeros(Np, dtype=dtype, device=device)
    x[0::2,0::2] = (1/2)*(a[0] + a[1] + a[2] + a[3])
    x[0::2,1::2] = (1/2)*(a[0] + a[1] - a[2] - a[3])
    x[1::2,0::2] = (1/2)*(a[0] - a[1] + a[2] - a[3])
    x[1::2,1::2] = (1/2)*(a[0] - a[1] - a[2] + a[3])
    return x

def _multi_level_haar2(x, Nlayers, dtype=torch.complex64, device='cpu'):
    if Nlayers == 1:
        return _haar2(x, dtype=dtype, device=device)
    else:
        a = _haar2(x, dtype=dtype, device=device)
        a_low = _multi_level_haar2(a[0], Nlayers - 1, dtype, device)
        return [a_low] + a[1:]
    
    
def _multi_level_ihaar2(a, Nlayers, dtype=torch.complex64, device='cpu'):
    if Nlayers == 1:
        return _ihaar2(a, dtype=dtype, device=device)
    else:
        a_low = _multi_level_ihaar2(a[0], Nlayers - 1, dtype, device)
        x = _ihaar2([a_low] + a[1:], dtype=dtype, device=device)
        return x

def _wavelet_block(a):
    a_top = torch.cat((a[0], a[1]), 0)
    a_bottom = torch.cat((a[2], a[3]), 0)
    return torch.cat((a_top, a_bottom), 1)

def _visualize(a):
    if len(a[0]) != 4:
        return _wavelet_block(a)
    else:
        a_lower = _visualize(a[0])
        return _wavelet_block([a_lower] + a[1:])
    
def visualize(x, Nlayers, dtype=torch.complex64, device='cpu'):
    with torch.no_grad():
        a = _multi_level_haar2(x, Nlayers, dtype=dtype, device=device)
        return _visualize(a)

class Haar2():
    def __init__(self, dtype=torch.float32, device='cpu'):        
        self.dtype = dtype
        
        self._check_inverse(device)
        
    def forward(self, x, shiftx=False, shifty=False, dtype=torch.complex64, device='cpu'):
        if shiftx: x = _roll(x,1)
        if shifty: x = _roll(x.permute(1,0),1)
        return _haar2(x, dtype, device)
    
    def inverse(self, a, shiftx=False, shifty=False, dtype=torch.complex64, device='cpu'):
        x = _ihaar2(a, dtype, device)
        if shifty: x = _roll(x,-1).permute(1,0)
        if shiftx: x = _roll(x,-1)
        return x
    
    def _check_inverse(self, device='cpu'):
        with torch.no_grad():
            x = torch.rand((256, 256), dtype=self.dtype, device=device)
            WHWx = self.inverse(self.forward(x, device=device), device=device)
            out = torch.sum(torch.abs(x - WHWx))
            assert out < EPS * x.shape[0] * x.shape[1], 'Inverse test failed!'

class HaarN():
    def __init__(self, Nlayers, dtype=torch.complex64, device='cpu'):
        self.Nlayer = Nlayer
        
        self.dtype = dtype
        
        self._check_inverse(device)
        
    def forward(self, x, shiftx=False, shifty=False, dtype=torch.complex64, device='cpu'):
        if shiftx: x = _roll(x,1)
        if shifty: x = _roll(x.permute(1,0),1)
        return _multi_level_haar2(x, self.Nlayers, dtype, device)
    
    def reverse(self, a, shiftx=False, shifty=False, dtype=torch.complex64, device='cpu'):
        x = _ihaar2(a, dtype, device)
        if shifty: x = _roll(x,-1).permute(1,0)
        if shiftx: x = _roll(x,-1)
        return x
    
    def _check_inverse(self, device='cpu'):
        with torch.no_grad():
            x = torch.rand((256, 256), dtype=self.dtype, device=device)
            WHWx = self.inverse(self.forward(x, device=device), device=device)
            out = torch.sum(torch.abs(x - WHWx))
            assert out < EPS * x.shape[0] * x.shape[1], 'Inverse test failed!'