# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import copy
import random
from abc import ABC, abstractmethod

from .data_set import Context, HDF5DataSet, DataSet, BigANNVectorDataSet
from .util import bulk_transform, parse_string_parameter, parse_int_parameter, \
    ConfigurationError


def register(registry):
    registry.register_param_source(
        "bulk-from-data-set", BulkVectorsFromDataSetParamSource
    )

    registry.register_param_source(
        "knn-query-from-data-set", QueryVectorsFromDataSetParamSource
    )

    registry.register_param_source(
        "knn-query-from-random", RandomQuerySource
    )


class RandomQuerySource:
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
        self.field_name = parse_string_parameter("field", params)
        self.k = parse_int_parameter("k", params)
        self.dimension = parse_int_parameter("dimension", params)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        vector = [random.random() for _ in range(self.dimension)]
        return {
            "index": self.index_name,
            "request-params": {
                "_source": {
                    "exclude": [self.field_name]
                }
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


class VectorsFromDataSetParamSource(ABC):
    """ Abstract class that can read vectors from a data set and partition the
    vectors across multiple clients.

    Attributes:
        index_name: Name of the index to generate the query for
        field_name: Name of the field to generate the query for
        data_set_format: Format data set is serialized with. bigann or hdf5
        data_set_path: Path to data set
        context: Context the data set will be used in.
        data_set: Structure containing meta data about data and ability to read
        num_vectors: Number of vectors to use from the data set
        total: Number of vectors for the partition
        current: Current vector offset in data set
        infinite: Property of param source signalling that it can be exhausted
        percent_completed: Progress indicator for how exhausted data set is
        offset: Offset into the data set to start at. Relevant when there are
                multiple partitions
    """
    def __init__(self, params, context: Context):
        self.index_name: str = parse_string_parameter("index", params)
        self.field_name: str = parse_string_parameter("field", params)

        self.context = context
        self.data_set_format = parse_string_parameter("data_set_format", params)
        self.data_set_path = parse_string_parameter("data_set_path", params)
        self.data_set: DataSet = self._read_data_set()

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
            return HDF5DataSet(self.data_set_path, self.context)
        if self.data_set_format == BigANNVectorDataSet.FORMAT_NAME:
            return BigANNVectorDataSet(self.data_set_path)
        raise ConfigurationError("Invalid data set format")

    def partition(self, partition_index, total_partitions):
        if self.num_vectors % total_partitions != 0:
            raise ValueError("Num vectors must be divisible by number of "
                             "partitions")

        partition_x = copy.copy(self)
        partition_x.num_vectors = int(self.num_vectors / total_partitions)
        partition_x.offset = int(partition_index * partition_x.num_vectors)

        # We need to create a new instance of the data set for each client
        partition_x.data_set = partition_x._read_data_set()
        partition_x.data_set.seek(partition_x.offset)
        partition_x.current = partition_x.offset
        return partition_x

    @abstractmethod
    def params(self):
        pass


class QueryVectorsFromDataSetParamSource(VectorsFromDataSetParamSource):
    """ Query parameter source for k-NN. Queries are created from data set
    provided.

    Attributes:
        k: The number of results to return for the search
        batch: List of vectors to be read from data set
    """

    VECTOR_READ_BATCH_SIZE = 100  # batch size to read vectors from data-set

    def __init__(self, workload, params, **kwargs):
        super().__init__(params, Context.QUERY)
        self.k = parse_int_parameter("k", params)
        self.batch = None

    def params(self):
        if self.current >= self.num_vectors + self.offset:
            raise StopIteration

        if self.batch is None or len(self.batch) == 0:
            self.batch = self._batch_read()
            if self.batch is None:
                raise StopIteration

        vector = self.batch.pop(0)
        self.current += 1
        self.percent_completed = self.current / self.total

        return {
            "index": self.index_name,
            "request-params": {
                "_source": {
                    "exclude": [self.field_name]
                }
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

    def _batch_read(self):
        return list(self.data_set.read(self.VECTOR_READ_BATCH_SIZE))


class BulkVectorsFromDataSetParamSource(VectorsFromDataSetParamSource):
    """ Create bulk index requests from a data set of vectors.

    Attributes:
        bulk_size: number of vectors per request
        retries: number of times to retry the request when it fails
    """
    def __init__(self, workload, params, **kwargs):
        super().__init__(params, Context.INDEX)
        self.bulk_size: int = parse_int_parameter("bulk_size", params)
        self.retries: int = parse_int_parameter("retries", params, 10)

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
