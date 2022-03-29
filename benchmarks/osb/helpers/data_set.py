# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import os
import numpy as np
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import cast
import h5py
import struct


class Context(Enum):
    """DataSet context enum. Can be used to add additional context for how a
    data-set should be interpreted.
    """
    INDEX = 1
    QUERY = 2
    NEIGHBORS = 3


class DataSet(ABC):
    """DataSet interface. Used for reading data-sets from files.

    Methods:
        read: Read a chunk of data from the data-set
        seek: Get to position in the data-set
        size: Gets the number of items in the data-set
        reset: Resets internal state of data-set to beginning
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, chunk_size: int):
        pass

    @abstractmethod
    def seek(self, offset: int):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class HDF5DataSet(DataSet):
    """ Data-set format corresponding to `ANN Benchmarks
    <https://github.com/erikbern/ann-benchmarks#data-sets>`_
    """

    def __init__(self, dataset_path: str, context: Context):
        file = h5py.File(dataset_path)
        self.data = cast(h5py.Dataset, file[self._parse_context(context)])
        self.current = 0

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = cast(np.ndarray, self.data[self.current:end_i])
        self.current = end_i
        return v

    def seek(self, offset: int):
        if offset >= self.size():
            raise Exception("Offset is greater than the size")

        self.current = offset

    def size(self):
        return self.data.len()

    def reset(self):
        self.current = 0

    @staticmethod
    def _parse_context(context: Context) -> str:
        if context == Context.NEIGHBORS:
            return "neighbors"

        if context == Context.INDEX:
            return "train"

        if context == Context.QUERY:
            return "test"

        raise Exception("Unsupported context")


class BigANNVectorDataSet(DataSet):
    """ Data-set format for vector data-sets for `Big ANN Benchmarks
    <https://big-ann-benchmarks.com/index.html#bench-datasets>`_
    """

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.file.seek(0, os.SEEK_END)
        num_bytes = self.file.tell()
        self.file.seek(0)

        if num_bytes < 8:
            raise Exception("File is invalid")

        self.num_points = int.from_bytes(self.file.read(4), "little")
        self.dimension = int.from_bytes(self.file.read(4), "little")
        bytes_per_num = self._get_data_size(dataset_path)

        if (num_bytes - 8) != self.num_points * self.dimension * bytes_per_num:
            raise Exception("File is invalid")

        self.reader = self._value_reader(dataset_path)
        self.current = 0

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = np.asarray([self._read_vector() for _ in
                        range(end_i - self.current)])
        self.current = end_i
        return v

    def seek(self, offset: int):
        if offset >= self.size():
            raise Exception("Offset is greater than the size")

        self.file.seek(offset)
        self.current = offset

    def _read_vector(self):
        return np.asarray([self.reader(self.file) for _ in
                           range(self.dimension)])

    def size(self):
        return self.num_points

    def reset(self):
        self.file.seek(8)  # Seek to 8 bytes to skip re-reading metadata
        self.current = 0

    @staticmethod
    def _get_data_size(file_name):
        ext = file_name.split('.')[-1]
        if ext == "u8bin":
            return 1

        if ext == "fbin":
            return 4

        raise Exception("Unknown extension")

    @staticmethod
    def _value_reader(file_name):
        ext = file_name.split('.')[-1]
        if ext == "u8bin":
            return lambda file: float(int.from_bytes(file.read(1), "little"))

        if ext == "fbin":
            return lambda file: struct.unpack('<f', file.read(4))

        raise Exception("Unknown extension")
