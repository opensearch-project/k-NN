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

    BEGINNING = 0

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

    FORMAT_NAME = "hdf5"

    def __init__(self, dataset_path: str, context: Context):
        file = h5py.File(dataset_path)
        self.data = cast(h5py.Dataset, file[self._parse_context(context)])
        self.current = self.BEGINNING

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_offset = self.current + chunk_size
        if end_offset > self.size():
            end_offset = self.size()

        v = cast(np.ndarray, self.data[self.current:end_offset])
        self.current = end_offset
        return v

    def seek(self, offset: int):

        if offset < self.BEGINNING:
            raise Exception("Offset must be greater than or equal to 0")

        if offset >= self.size():
            raise Exception("Offset must be less than the data set size")

        self.current = offset

    def size(self):
        return self.data.len()

    def reset(self):
        self.current = self.BEGINNING

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

    DATA_SET_HEADER_LENGTH = 8
    U8BIN_EXTENSION = "u8bin"
    FBIN_EXTENSION = "fbin"
    FORMAT_NAME = "bigann"

    BYTES_PER_U8INT = 1
    BYTES_PER_FLOAT = 4

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.file.seek(BigANNVectorDataSet.BEGINNING, os.SEEK_END)
        num_bytes = self.file.tell()
        self.file.seek(BigANNVectorDataSet.BEGINNING)

        if num_bytes < BigANNVectorDataSet.DATA_SET_HEADER_LENGTH:
            raise Exception("File is invalid")

        self.num_points = int.from_bytes(self.file.read(4), "little")
        self.dimension = int.from_bytes(self.file.read(4), "little")
        self.bytes_per_num = self._get_data_size(dataset_path)

        if (num_bytes - BigANNVectorDataSet.DATA_SET_HEADER_LENGTH) != self.num_points * \
                self.dimension * self.bytes_per_num:
            raise Exception("File is invalid")

        self.reader = self._value_reader(dataset_path)
        self.current = BigANNVectorDataSet.BEGINNING

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_offset = self.current + chunk_size
        if end_offset > self.size():
            end_offset = self.size()

        v = np.asarray([self._read_vector() for _ in
                        range(end_offset - self.current)])
        self.current = end_offset
        return v

    def seek(self, offset: int):

        if offset < self.BEGINNING:
            raise Exception("Offset must be greater than or equal to 0")

        if offset >= self.size():
            raise Exception("Offset must be less than the data set size")

        bytes_offset = BigANNVectorDataSet.DATA_SET_HEADER_LENGTH + \
                       self.dimension * self.bytes_per_num * offset
        self.file.seek(bytes_offset)
        self.current = offset

    def _read_vector(self):
        return np.asarray([self.reader(self.file) for _ in
                           range(self.dimension)])

    def size(self):
        return self.num_points

    def reset(self):
        self.file.seek(BigANNVectorDataSet.DATA_SET_HEADER_LENGTH)
        self.current = BigANNVectorDataSet.BEGINNING

    @staticmethod
    def _get_data_size(file_name):
        ext = file_name.split('.')[-1]
        if ext == BigANNVectorDataSet.U8BIN_EXTENSION:
            return BigANNVectorDataSet.BYTES_PER_U8INT

        if ext == BigANNVectorDataSet.FBIN_EXTENSION:
            return BigANNVectorDataSet.BYTES_PER_FLOAT

        raise Exception("Unknown extension")

    @staticmethod
    def _value_reader(file_name):
        ext = file_name.split('.')[-1]
        if ext == BigANNVectorDataSet.U8BIN_EXTENSION:
            return lambda file: float(int.from_bytes(file.read(BigANNVectorDataSet.BYTES_PER_U8INT), "little"))

        if ext == BigANNVectorDataSet.FBIN_EXTENSION:
            return lambda file: struct.unpack('<f', file.read(BigANNVectorDataSet.BYTES_PER_FLOAT))

        raise Exception("Unknown extension")


class BigANNNeighborDataSet(DataSet):
    """ Data-set format for neighbor data-sets for `Big ANN Benchmarks
    <https://big-ann-benchmarks.com/index.html#bench-datasets>`_"""

    DATA_SET_HEADER_LENGTH = 8
    INT_32_SIZE = 4
    FLOAT_32_SIZE = 4

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.file.seek(BigANNNeighborDataSet.BEGINNING, os.SEEK_END)
        num_bytes = self.file.tell()
        self.file.seek(BigANNNeighborDataSet.BEGINNING)

        if num_bytes < BigANNNeighborDataSet.DATA_SET_HEADER_LENGTH:
            raise Exception("File is invalid")

        self.num_queries = int.from_bytes(
            self.file.read(BigANNNeighborDataSet.INT_32_SIZE), "little")
        self.k = int.from_bytes(
            self.file.read(BigANNNeighborDataSet.INT_32_SIZE), "little")

        # According to the website, the number of bytes that will follow will
        # be:  num_queries X K x sizeof(uint32_t) bytes + num_queries X K x
        # sizeof(float)
        if (num_bytes - BigANNNeighborDataSet.DATA_SET_HEADER_LENGTH) != 2 * \
                (self.num_queries * self.k *
                 BigANNNeighborDataSet.FLOAT_32_SIZE):
            raise Exception("File is invalid")

        self.current = BigANNNeighborDataSet.BEGINNING

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = [[int.from_bytes(
                    self.file.read(BigANNNeighborDataSet.INT_32_SIZE), "little")
                    for _ in range(self.k)] for _ in range(end_i - self.current)
        ]

        self.current = end_i
        return v

    def seek(self, offset: int):

        if offset < BigANNNeighborDataSet.BEGINNING:
            raise Exception("Offset must be greater than or equal to 0")

        if offset >= self.size():
            raise Exception("Offset must be less than the data set size")

        # Go to the offset set of nearest neighbors
        bytes_offset = BigANNNeighborDataSet.DATA_SET_HEADER_LENGTH + self.k * \
                       BigANNNeighborDataSet.INT_32_SIZE * offset
        self.file.seek(bytes_offset)
        self.current = offset

    def size(self):
        return self.num_queries

    def reset(self):
        self.file.seek(BigANNNeighborDataSet.DATA_SET_HEADER_LENGTH)
        self.current = BigANNNeighborDataSet.BEGINNING
