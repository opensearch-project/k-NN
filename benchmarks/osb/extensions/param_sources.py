# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import copy
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
        self.data_set: DataSet = self._read_data_set(self.data_set_format,
                                                     self.data_set_path,
                                                     self.context)

        self.num_vectors: int = parse_int_parameter(
            "num_vectors", params, self.data_set.size()
        )
        self.total = self.num_vectors
        self.current = 0
        self.infinite = False
        self.percent_completed = 0
        self.offset = 0

    def _read_data_set(self, data_set_format: str, data_set_path: str,
                       data_set_context: Context):
        if data_set_format == HDF5DataSet.FORMAT_NAME:
            return HDF5DataSet(data_set_path, data_set_context)
        if data_set_format == BigANNVectorDataSet.FORMAT_NAME:
            return BigANNVectorDataSet(data_set_path)
        raise ConfigurationError("Invalid data set format")

    def partition(self, partition_index, total_partitions):
        """
        Splits up the parameters source so that multiple clients can read data
        from it.
        Args:
            partition_index: index of one particular partition
            total_partitions: total number of partitions data set is split into

        Returns:
            The parameter source for this particular partion
        """
        if self.num_vectors % total_partitions != 0:
            raise ValueError("Num vectors must be divisible by number of "
                             "partitions")

        partition_x = copy.copy(self)
        partition_x.num_vectors = int(self.num_vectors / total_partitions)
        partition_x.offset = int(partition_index * partition_x.num_vectors)

        # We need to create a new instance of the data set for each client
        partition_x.data_set = partition_x._read_data_set(
            self.data_set_format,
            self.data_set_path,
            self.context
        )
        partition_x.data_set.seek(partition_x.offset)
        partition_x.current = partition_x.offset
        return partition_x

    @abstractmethod
    def params(self):
        """
        Returns: A single parameter from this sourc
        """
        pass


class QueryVectorsFromDataSetParamSource(VectorsFromDataSetParamSource):
    """ Query parameter source for k-NN. Queries are created from data set
    provided.

    Attributes:
        k: The number of results to return for the search
        vector_batch: List of vectors to be read from data set. Read are batched
                        so that we do not need to read from disk for each query
    """

    VECTOR_READ_BATCH_SIZE = 100  # batch size to read vectors from data-set

    def __init__(self, workload, params, **kwargs):
        super().__init__(params, Context.QUERY)
        self.k = parse_int_parameter("k", params)
        self.vector_batch = None

    def params(self):
        """
        Returns: A query parameter with a vector from a data set
        """
        if self.current >= self.num_vectors + self.offset:
            raise StopIteration

        if self.vector_batch is None or len(self.vector_batch) == 0:
            self.vector_batch = self._batch_read(self.data_set)
            if self.vector_batch is None:
                raise StopIteration
        vector = self.vector_batch.pop(0)
        self.current += 1
        self.percent_completed = self.current / self.total

        return self._build_query_body(self.index_name, self.field_name, self.k,
                                      vector)

    def _batch_read(self, data_set: DataSet):
        return list(data_set.read(self.VECTOR_READ_BATCH_SIZE))

    def _build_query_body(self, index_name: str, field_name: str, k: int,
                          vector) -> dict:
        """Builds a k-NN query that can be used to execute an approximate nearest
        neighbor search against a k-NN plugin index
        Args:
            index_name: name of index to search
            field_name: name of field to search
            k: number of results to return
            vector: vector used for query
        Returns:
            A dictionary containing the body used for search, a set of request
            parameters to attach to the search and the name of the index.
        """
        return {
            "index": index_name,
            "request-params": {
                "_source": {
                    "exclude": [field_name]
                }
            },
            "body": {
                "size": k,
                "query": {
                    "knn": {
                        field_name: {
                            "vector": vector,
                            "k": k
                        }
                    }
                }
            }
        }


class BulkVectorsFromDataSetParamSource(VectorsFromDataSetParamSource):
    """ Create bulk index requests from a data set of vectors.

    Attributes:
        bulk_size: number of vectors per request
        retries: number of times to retry the request when it fails
    """

    DEFAULT_RETRIES = 10

    def __init__(self, workload, params, **kwargs):
        super().__init__(params, Context.INDEX)
        self.bulk_size: int = parse_int_parameter("bulk_size", params)
        self.retries: int = parse_int_parameter("retries", params,
                                                self.DEFAULT_RETRIES)

    def params(self):
        """
        Returns: A bulk index parameter with vectors from a data set.
        """
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
