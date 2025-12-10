from __future__ import annotations
import torch
import numpy as np
import json
from exllamav2.ext import none_tensor, exllamav2_ext as ext_c
import os

def convert_dtype(dt: str):
    """
    :param dt:
        Datatype string as used by safetensors

    :return:
        Torch type, element size in bytes
    """
    if dt == "I32": return torch.int, 4
    elif dt == "I16": return torch.short, 2
    elif dt == "F16": return torch.float16, 2
    elif dt == "BF16": return torch.bfloat16, 2
    elif dt == "F32": return torch.float, 4
    else:
        raise ValueError(f"Unknown dtype {dt}")

global_stfiles = {}

def cleanup_stfiles():
    """
    Close all file handles and free any storage used while loading tensors
    """
    global global_stfiles
    # for stf in global_stfiles:
    #     stf.close()
    global_stfiles = {}
    # ext_c.stfiles_free_pinned_buffer()

class STFile:

    filename: str
    header: dict
    header_size: int
    metadata: object
    handle: int
    tensor_remap: dict | None

    def __init__(
        self,
        filename: str,
        keymap: list[tuple[str, str]] = None
    ):
        self.filename = filename
        self.metadata = None
        self.handle = 0
        self.tensor_remap = None
        self.read_dict()
        if keymap:
            self.remap_dict(keymap)

    @staticmethod
    def open(
        filename,
        keymap: list[tuple[str, str]] = None
    ) -> STFile:
        """
        Open safetensors file, scan header and retain handle.

        :param filename:
            Filename

        :param keymap:
            List of (a, b) tuples for string replacements in key index

        :return:
            STFile object
        """
        global global_stfiles
        if filename not in global_stfiles:
            global_stfiles[filename] = STFile(filename, keymap)
        return global_stfiles[filename]

    def close(self):
        """
        Close file handle (if necessary)
        """
        assert self.filename in global_stfiles, \
            f"Can't close {self.filename}: already closed"
        # if self.fast_fb: ext_c.safetensors_close_fb(self.handle_fb)

    def read_dict(self):
        with open(self.filename, "rb") as fp:
            header_size = np.fromfile(fp, dtype = np.int64, count = 1).item()
            header_json = fp.read(header_size)
            self.header = json.loads(header_json.decode("utf-8"))
            self.header_size = fp.tell()
            if "__metadata__" in self.header:
                self.metadata = self.header["__metadata__"]
                del self.header["__metadata__"]

    def remap_dict(self, keymap: list[tuple[str, str]]):
        self.tensor_remap = {}
        nheader = {}
        for key in self.header.keys():
            nkey = key
            for z in keymap:
                if z[0].startswith("$") and nkey.startswith(z[0][1:]):
                    nkey = ("$" + nkey).replace(z[0], z[1])
                else:
                    nkey = nkey.replace(z[0], z[1])
            nheader[nkey] = self.header[key]
            self.tensor_remap[nkey] = key
        self.header = nheader

    def get_dict(self) -> dict:
        return self.header

    def get_metadata(self) -> object:
        return self.metadata

    def measure(self, key):
        """
        :param key:
            Tensor key

        :return:
            Byte size of tensor
        """
        v = self.header[key]
        data_offsets = v["data_offsets"]
        length = data_offsets[1] - data_offsets[0]
        return length

    def get_tensor(
        self,
        key: str,
        device,
        out_dtype = None
    ) -> torch.Tensor:
        """
        Load tensor from file

        :param key:
            Tensor name

        :param device:
            Target device

        :param out_dtype:
            Force output datatype

        :return:
            torch.Tensor
        """
        h = self.header[key]
        dtype, esize = convert_dtype(h["dtype"])
        beg, end = h["data_offsets"]
        size = end - beg
        shape = h["shape"]
        tensor = torch.zeros(shape, dtype = dtype, device = device)
        torch.cuda.synchronize()
        assert tensor.is_contiguous, "Non-contiguous tensor"
        ext_c.stloader_read(
            self.filename,
            beg + self.header_size,
            size,
            tensor
        )
        if out_dtype:
            tensor = tensor.to(out_dtype)
        return tensor

    def get_tensors(
        self,
        keys: list,
        device,
        out_dtypes = None
    ) -> dict:
        """
        Batch load multiple tensors from file.

        :param keys:
            List of tensor names

        :param device:
            Target device

        :param out_dtypes:
            Optional list of output dtypes (or None for each)

        :return:
            dict of {key: tensor}
        """
        tensors = {}
        if out_dtypes is None:
            out_dtypes = [None] * len(keys)
        for key, out_dtype in zip(keys, out_dtypes):
            h = self.header[key]
            dtype, esize = convert_dtype(h["dtype"])
            beg, end = h["data_offsets"]
            size = end - beg
            shape = h["shape"]
            tensor = torch.empty(shape, dtype = dtype, device = device)
            torch.cuda.synchronize()
            assert tensor.is_contiguous, "Non-contiguous tensor"
            ext_c.stloader_read(
                self.filename,
                beg + self.header_size,
                size,
                tensor
            )
            if out_dtype:
                tensor = tensor.to(out_dtype)
            tensors[key] = tensor
        return tensors
