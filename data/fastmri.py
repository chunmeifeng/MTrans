import csv
import os

import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from .transforms import build_transforms
from matplotlib import pyplot as plt


def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            challenge,
            sample_rate=1,
            mode='train'
    ):
        self.mode = mode

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        # transform
        self.transform = transform

        self.examples = []

        self.cur_path = root
        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            id = 0

            for row in reader:
                pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[0] + '.h5'))

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1] + '.h5'))

                for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                    self.examples.append(
                        (os.path.join(self.cur_path, row[0] + '.h5'), os.path.join(self.cur_path, row[1] + '.h5')
                         , slice_id, pd_metadata, pdfs_metadata, id))
                id += 1

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        # 读取pd
        pd_fname, pdfs_fname, slice, pd_metadata, pdfs_metadata, id = self.examples[i]

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]

            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pd_metadata)

        if self.transform is None:
            pd_sample = (pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)
        else:
            pd_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)

        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][slice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None

            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(pdfs_metadata)

        if self.transform is None:
            pdfs_sample = (pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)
        else:
            pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)

        # vis_data(pdfs_sample[0], pdfs_target[0], pd_fname, pdfs_fname, slice, 'vis_noise')

        return (pd_sample, pdfs_sample, id)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices


def build_dataset(args, mode='train', sample_rate=1):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode)
    return SliceDataset(os.path.join(args.DATASET.ROOT, 'singlecoil_' + mode), transforms, args.DATASET.CHALLENGE,
                        sample_rate=sample_rate, mode=mode)
