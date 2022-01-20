# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Defines DataSet interface and implements particular formats

A DataSet is thas the basic functionality that it can be read in chunks, or
read completely and reset to the start.

Currently, we support HDF5 formats from ann-benchmarks and big-ann-benchmarks
datasets.

Classes:
    HDF5DataSet: Format used in ann-benchmarks
    BigANNNeighborDataSet: Neighbor format for big-ann-benchmarks
    BigANNVectorDataSet: Vector format for big-ann-benchmarks
"""

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import cast
import h5py
import numpy as np

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
        size: Gets the number of items in the data-set
        reset: Resets internal state of data-set to beginning
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, chunk_size: int):
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


class BigANNNeighborDataSet(DataSet):
    """ Data-set format for neighbor data-sets for `Big ANN Benchmarks
    <https://big-ann-benchmarks.com/index.html#bench-datasets>`_"""

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.num_queries = int.from_bytes(self.file.read(4), "little")
        self.k = int.from_bytes(self.file.read(4), "little")
        self.current = 0

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = [[int.from_bytes(self.file.read(4), "little") for _ in
              range(self.k)] for _ in range(end_i - self.current)]

        self.current = end_i
        return v

    def size(self):
        return self.num_queries

    def reset(self):
        self.file.seek(8)
        self.current = 0


class BigANNVectorDataSet(DataSet):
    """ Data-set format for vector data-sets for `Big ANN Benchmarks
    <https://big-ann-benchmarks.com/index.html#bench-datasets>`_
    """

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.num_points = int.from_bytes(self.file.read(4), "little")
        self.dimension = int.from_bytes(self.file.read(4), "little")
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

    def _read_vector(self):
        return np.asarray([self.reader(self.file) for _ in
                           range(self.dimension)])

    def size(self):
        return self.num_points

    def reset(self):
        self.file.seek(8)  # Seek to 8 bytes to skip re-reading metadata
        self.current = 0

    @staticmethod
    def _value_reader(file_name):
        ext = file_name.split('.')[-1]
        if ext == "u8bin":
            return lambda file: float(int.from_bytes(file.read(1), "little"))

        if ext == "fbin":
            return lambda file: struct.unpack('<f', file.read(4))

        raise Exception("Unknown extension")
