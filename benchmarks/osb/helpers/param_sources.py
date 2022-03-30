# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from helpers.data_set import Context, HDF5DataSet, DataSet, \
    BigANNVectorDataSet
from helpers.helper import bulk_transform


def register(registry):
    registry.register_param_source("bulk-from-data-set",
                                   BulkVectorsFromDataSetParamSource)


class BulkVectorsFromDataSetParamSource:
    def __init__(self, workload, params, **kwargs):
        self.data_set: DataSet
        data_set_format = params["data_set_format"]
        data_set_path = params["data_set_path"]

        if data_set_format == "hdf5":
            self.data_set = HDF5DataSet(data_set_path, Context.INDEX)
        elif data_set_format == "bigann":
            self.data_set = BigANNVectorDataSet(data_set_path)

        self.field_name: str = params["field"]
        self.index_name: str = params["index"]
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

