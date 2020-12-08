# Wavelet Transform for Pytorch

This package provides a differentiable Pytorch implementation of the Haar wavelet transform.

![asdf](https://github.com/kellman/pytorch_wavelet/blob/main/figures/output.png)

## Usage

``` 
import torch
import matplotlib.pyplot as plt
from skimage import data
import pytorch_wavelet as wavelet

x = torch.from_numpy(data.camera())
a = wavelet.visualize(x, Nlayers = 2)

plt.figure()
plt.subplot(121)
plt.imshow(x)
plt.title('Image')
plt.subplot(122)
plt.imshow(a)
plt.title('Wavelet Decomposition')

```

## Install

```pip install pytorch-wavelet ```

