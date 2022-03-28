# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import os
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import cast
import h5py
import numpy as np
from typing import List
from typing import Dict
from typing import Any
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


def bulk_transform(partition: np.ndarray, field_name: str, action,
                   offset: int) -> List[Dict[str, Any]]:
    """Partitions and transforms a list of vectors into OpenSearch's bulk
    injection format.
    Args:
        offset: to start counting from
        partition: An array of vectors to transform.
        field_name: field name for action
        action: Bulk API action.
    Returns:
        An array of transformed vectors in bulk format.
    """
    actions = []
    _ = [
        actions.extend([action(i + offset), None])
        for i in range(len(partition))
    ]
    actions[1::2] = [{field_name: vec} for vec in partition.tolist()]
    return actions


class BulkVectorsFromDataSetParamSource:
    def __init__(self, workload, params, **kwargs):
        self.data_set: DataSet
        data_set_format = params["data_set_format"]
        data_set_path = params["data_set_path"]

        if data_set_format == "hdf5":
            self.data_set = HDF5DataSet(data_set_path, Context.INDEX)
        elif data_set_format == "bigann":
            self.data_set = BigANNVectorDataSet(data_set_path)

        self.field_name: str = params["field_name"]
        self.index_name: str = params["index_name"]
        self.bulk_size: int = params["bulk_size"]
        self.current = 0
        self.infinite = False
        self.percent_completed = 0
        self.total = self.data_set.size()
        self.offset = 0

    def partition(self, partition_index, total_partitions):
        if self.data_set.size() % total_partitions != 0:
            raise Exception("Data set must be divisible by number of clients")

        partition_x = self
        partition_x.total = int(self.data_set.size() / total_partitions)
        partition_x.offset = int(partition_index * partition_x.total)
        partition_x.data_set.seek(partition_x.offset)
        partition_x.current = partition_x.offset
        return partition_x

    def params(self):

        if self.current >= self.total + self.offset:
            raise StopIteration

        def action(doc_id):
            return {'index': {'_index': self.index_name, '_id': doc_id}}

        partition = self.data_set.read(self.bulk_size)
        body = bulk_transform(partition, self.field_name, action, self.current)
        self.current += int(len(body) / 2)
        self.percent_completed = float(self.current)/self.data_set.size()

        return body


class BulkVectorsFromDataSetRunner:

    async def __call__(self, opensearch, params):
        attempts = 10  # TODO: parametrize this
        for _ in range(attempts):
            try:
                await opensearch.bulk(
                    body=params,
                    timeout='5m'
                )
            except:
                pass

    def __repr__(self, *args, **kwargs):
        return "custom-vector-bulk"


class CustomRefreshRunner:

    async def __call__(self, opensearch, params):
        # Basically just keep calling it until it succeeds
        attempts = params["retries"]

        for _ in range(attempts):
            try:
                await opensearch.indices.refresh(index=params["index_name"])
            except:
                pass

    def __repr__(self, *args, **kwargs):
        return "custom-refresh"


def register(registry):
    registry.register_param_source("bulk-from-data-set", BulkVectorsFromDataSetParamSource)
    registry.register_runner("custom-vector-bulk", BulkVectorsFromDataSetRunner(), async_runner=True)
    registry.register_runner("custom-refresh", CustomRefreshRunner(), async_runner=True)
