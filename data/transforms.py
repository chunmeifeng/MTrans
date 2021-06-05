"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

from .math import ifft2c, fft2c, complex_abs
from .subsample import create_mask_for_mask_type, MaskFunc
import random

from typing import Dict, Optional, Sequence, Tuple, Union
from matplotlib import pyplot as plt
import os

def rss(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor.
    
    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data (np.array): Input numpy array.

    Returns:
        torch.Tensor: PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(data, mask_func, seed=None, padding=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


def mask_center(x, mask_from, mask_to):
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than
            the corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image.
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It
            should have at least 3 dimensions and the cropping is applied along
            dimensions -3 and -2 and the last dimensions should have a size of
            2.
        shape (int): The output shape. The shape should be smaller than
            the corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]

    w_from = (data.shape[-3] - shape[0]) // 2   #80
    h_from = (data.shape[-2] - shape[1]) // 2   #80
    w_to = w_from + shape[0]  #240
    h_to = h_from + shape[1]  #240

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(x, y):
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.
    
    Args:
        x (torch.Tensor): The first image.
        y (torch.Tensor): The second image

    Returns:
        tuple: tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data (torch.Tensor): Input data to be normalized.
        mean (float): Mean value.
        stddev (float): Standard deviation.
        eps (float, default=0.0): Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class DataTransform(object):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, which_challenge):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.which_challenge = which_challenge

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            mask (numpy.array): Mask from the test dataset.
            target (numpy.array): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # inverse Fourier transform to get zero filled solution
        image = ifft2c(kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for sFLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # getLR
        imgfft = fft2c(image)
        imgfft = complex_center_crop(imgfft, (160, 160))
        LR_image = ifft2c(imgfft)

        # absolute value
        LR_image = complex_abs(LR_image)

        # normalize input
        LR_image, mean, std = normalize_instance(LR_image, eps=1e-11)
        LR_image = LR_image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return LR_image, target, mean, std, fname, slice_num

class DenoiseDataTransform(object):
    def __init__(self, size, noise_rate):
        super(DenoiseDataTransform, self).__init__()
        self.size = (size, size)
        self.noise_rate = noise_rate
    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        max_value = attrs["max"]

        #target
        target = to_tensor(target)
        target = center_crop(target, self.size)
        target, mean, std = normalize_instance(target, eps=1e-11)
        target = target.clamp(-6, 6)

        #image
        kspace = to_tensor(kspace)
        complex_image = ifft2c(kspace)     #complex_image
        image = complex_center_crop(complex_image, self.size)
        noise_image = self.rician_noise(image, max_value)
        noise_image = complex_abs(noise_image)

        noise_image = normalize(noise_image, mean, std, eps=1e-11)
        noise_image = noise_image.clamp(-6, 6)

        return noise_image, target, mean, std, fname, slice_num


    def rician_noise(self, X, noise_std):
        #Add rician noise with variance sampled uniformly from the range 0 and 0.1
        noise_std = random.uniform(0, noise_std*self.noise_rate)
        Ir = X + noise_std * torch.randn(X.shape)
        Ii = noise_std*torch.randn(X.shape)
        In = torch.sqrt(Ir ** 2 + Ii ** 2)
        return In


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask



class ReconstructionTransform(object):
    """
       Data Transformer for training U-Net models.
       """

    def __init__(self, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            mask (numpy.array): Mask from the test dataset.
            target (numpy.array): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for sFLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)
        # print('image',image.shape)
        # absolute value
        image = complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num


def build_transforms(args, mode = 'train'):

    if args.WORK_TYPE == 'sr':

        if mode == 'train':
            return DataTransform(args.DATASET.CHALLENGE)
        elif mode == 'val':
            return DataTransform(args.DATASET.CHALLENGE)
        else:
            return DataTransform(args.DATASET.CHALLENGE)

    elif args.WORK_TYPE == 'denoise':
        return DenoiseDataTransform(args.INPUT_SIZE, args.NOISE_RATE)

    else:
        if mode == 'train':
            mask = create_mask_for_mask_type(
                args.TRANSFORMS.MASKTYPE, args.TRANSFORMS.CENTER_FRACTIONS, args.TRANSFORMS.ACCELERATIONS,
            )
            return ReconstructionTransform(args.DATASET.CHALLENGE, mask, use_seed=False)
        elif mode == 'val':
            mask = create_mask_for_mask_type(
                args.TRANSFORMS.MASKTYPE, args.TRANSFORMS.CENTER_FRACTIONS, args.TRANSFORMS.ACCELERATIONS,
            )
            return ReconstructionTransform(args.DATASET.CHALLENGE, mask)
        else:
            return ReconstructionTransform(args.DATASET.CHALLENGE)





