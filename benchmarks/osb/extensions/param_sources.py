# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from .data_set import Context, HDF5DataSet, DataSet, BigANNVectorDataSet
from .util import bulk_transform, parse_string_parameter, parse_int_parameter


def register(registry):
    registry.register_param_source(
        "bulk-from-data-set", BulkVectorsFromDataSetParamSource
    )


class BulkVectorsFromDataSetParamSource:
    def __init__(self, workload, params, **kwargs):
        self.data_set: DataSet
        data_set_format = parse_string_parameter("data_set_format", params)
        data_set_path = parse_string_parameter("data_set_path", params)

        if data_set_format == "hdf5":
            self.data_set = HDF5DataSet(data_set_path, Context.INDEX)
        elif data_set_format == "bigann":
            self.data_set = BigANNVectorDataSet(data_set_path)

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

    def partition(self, partition_index, total_partitions):
        if self.data_set.size() % total_partitions != 0:
            raise Exception("Data set must be divisible by number of clients")

        partition_x = self
        partition_x.num_vectors = int(self.num_vectors / total_partitions)
        partition_x.offset = int(partition_index * partition_x.num_vectors)
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
        self.current += int(len(body) / 2)
        self.percent_completed = float(self.current)/self.total

        return {
            "body": body,
            "retries": self.retries
        }
