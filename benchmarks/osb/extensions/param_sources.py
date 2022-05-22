# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import copy
import random

from .data_set import Context, HDF5DataSet, DataSet, BigANNVectorDataSet
from .util import bulk_transform, parse_string_parameter, parse_int_parameter, \
    ConfigurationError


def register(registry):
    registry.register_param_source(
        "bulk-from-data-set", BulkVectorsFromDataSetParamSource
    )

    registry.register_param_source(
        "random-knn-query", RandomKNNQuerySource
    )


class RandomKNNQuerySource:
    """ Query parameter source for k-NN. Queries are randomly generated. Can be
    configured.

    Attributes:
        index_name: Name of the index to generate the query for
        field_name: Name of the field to generate the query for
        k: The number of results to return for the search
        dimension: Dimension of vectors to produce
    """
    def __init__(self, workload, params, **kwargs):
        self.index_name = parse_string_parameter("index", params)
        self.field_name = parse_string_parameter("field_name", params)
        self.k = parse_int_parameter("k", params)
        self.dimension = parse_int_parameter("dimension", params)

    def params(self):
        vector = [random.random() for _ in range(self.dimension)]
        return {
            "index": self.index_name,
            "request-params": {
                "exclude": [self.field_name]
            },
            "body": {
                "size": self.k,
                "query": {
                    "knn": {
                        self.field_name: {
                            "vector": vector,
                            "k": self.k
                        }
                    }
                }
            }
        }


class BulkVectorsFromDataSetParamSource:
    def __init__(self, workload, params, **kwargs):
        self.data_set_format = parse_string_parameter("data_set_format", params)
        self.data_set_path = parse_string_parameter("data_set_path", params)
        self.data_set: DataSet = self._read_data_set()

        self.field_name: str = parse_string_parameter("field", params)
        self.index_name: str = parse_string_parameter("index", params)
        self.bulk_size: int = parse_int_parameter("bulk_size", params)
        self.retries: int = parse_int_parameter("retries", params, 10)
        self.num_vectors: int = parse_int_parameter(
            "num_vectors", params, self.data_set.size()
        )
        self.total = self.num_vectors
        self.current = 0
        self.infinite = False
        self.percent_completed = 0
        self.offset = 0

    def _read_data_set(self):
        if self.data_set_format == HDF5DataSet.FORMAT_NAME:
            return HDF5DataSet(self.data_set_path, Context.INDEX)
        if self.data_set_format == BigANNVectorDataSet.FORMAT_NAME:
            return BigANNVectorDataSet(self.data_set_path)
        raise ConfigurationError("Invalid data set format")

    def partition(self, partition_index, total_partitions):
        if self.data_set.size() % total_partitions != 0:
            raise ValueError("Data set must be divisible by number of clients")

        partition_x = copy.copy(self)
        partition_x.num_vectors = int(self.num_vectors / total_partitions)
        partition_x.offset = int(partition_index * partition_x.num_vectors)

        # We need to create a new instance of the data set for each client
        partition_x.data_set = partition_x._read_data_set()
        partition_x.data_set.seek(partition_x.offset)
        partition_x.current = partition_x.offset
        return partition_x

    def params(self):

        if self.current >= self.num_vectors + self.offset:
            raise StopIteration

        def action(doc_id):
            return {'index': {'_index': self.index_name, '_id': doc_id}}

        partition = self.data_set.read(self.bulk_size)
        body = bulk_transform(partition, self.field_name, action, self.current)
        size = len(body) // 2
        self.current += size
        self.percent_completed = self.current / self.total

        return {
            "body": body,
            "retries": self.retries,
            "size": size
        }
