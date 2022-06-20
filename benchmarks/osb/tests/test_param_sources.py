# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import os
import random
import shutil
import string
import sys
import tempfile
import unittest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from osb.tests.data_set_helper import HDF5Builder, create_random_2d_array, \
    DataSetBuildContext, BigANNBuilder
from osb.extensions.data_set import Context, HDF5DataSet
from osb.extensions.param_sources import VectorsFromDataSetParamSource, \
    QueryVectorsFromDataSetParamSource, BulkVectorsFromDataSetParamSource
from osb.extensions.util import ConfigurationError

DEFAULT_INDEX_NAME = "test-index"
DEFAULT_FIELD_NAME = "test-field"
DEFAULT_CONTEXT = Context.INDEX
DEFAULT_TYPE = HDF5DataSet.FORMAT_NAME


class VectorsFromDataSetParamSourceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.data_set_dir = tempfile.mkdtemp()

        # Create a data set we know to be valid for convenience
        self.valid_data_set_path = _create_data_set(
            10,
            10,
            DEFAULT_TYPE,
            DEFAULT_CONTEXT,
            self.data_set_dir
        )

    def tearDown(self):
        shutil.rmtree(self.data_set_dir)

    def test_missing_params(self):
        empty_params = dict()
        self.assertRaises(
            ConfigurationError,
            lambda: VectorsFromDataSetParamSourceTestCase.
                TestVectorsFromDataSetParamSource(empty_params, DEFAULT_CONTEXT)
        )

    def test_invalid_data_set_format(self):
        invalid_data_set_format = "invalid-data-set-format"

        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": invalid_data_set_format,
            "data_set_path": self.valid_data_set_path,
        }
        self.assertRaises(
            ConfigurationError,
            lambda: self.TestVectorsFromDataSetParamSource(
                test_param_source_params,
                DEFAULT_CONTEXT
            )
        )

    def test_invalid_data_set_path(self):
        invalid_data_set_path = "invalid-data-set-path"
        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": HDF5DataSet.FORMAT_NAME,
            "data_set_path": invalid_data_set_path,
        }
        self.assertRaises(
            FileNotFoundError,
            lambda: self.TestVectorsFromDataSetParamSource(
                test_param_source_params,
                DEFAULT_CONTEXT
            )
        )

    def test_partition_hdf5(self):

        hdf5_data_set_path = _create_data_set(
            100,
            10,
            HDF5DataSet.FORMAT_NAME,
            DEFAULT_CONTEXT,
            self.data_set_dir
        )

        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": HDF5DataSet.FORMAT_NAME,
            "data_set_path": hdf5_data_set_path,
        }
        test_param_source = self.TestVectorsFromDataSetParamSource(
            test_param_source_params,
            DEFAULT_CONTEXT
        )

        num_partitions = 10
        vecs_per_partition = test_param_source.num_vectors // num_partitions

        self._test_partition(
            test_param_source,
            num_partitions,
            vecs_per_partition
        )

    def test_partition_bigann(self):

        bigann_data_set_path = _create_data_set(
            100,
            10,
            "fbin",
            DEFAULT_CONTEXT,
            self.data_set_dir
        )

        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": "bigann",
            "data_set_path": bigann_data_set_path,
        }
        test_param_source = self.TestVectorsFromDataSetParamSource(
            test_param_source_params,
            DEFAULT_CONTEXT
        )

        num_partitions = 10
        vecs_per_partition = test_param_source.num_vectors // num_partitions

        self._test_partition(
            test_param_source,
            num_partitions,
            vecs_per_partition
        )

    def _test_partition(
            self,
            test_param_source: VectorsFromDataSetParamSource,
            num_partitions: int,
            vec_per_partition: int
    ):
        for i in range(num_partitions):
            test_param_source_i = test_param_source.partition(i, num_partitions)
            self.assertEqual(test_param_source_i.num_vectors, vec_per_partition)
            self.assertEqual(test_param_source_i.offset, i * vec_per_partition)

    class TestVectorsFromDataSetParamSource(VectorsFromDataSetParamSource):
        """
        Empty implementation of ABC VectorsFromDataSetParamSource so that we can
        test the concrete methods.
        """

        def params(self):
            pass


class QueryVectorsFromDataSetParamSourceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.data_set_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.data_set_dir)

    def test_params(self):
        # Create a data set
        num_vectors = 10
        dimension = 10
        k = 12

        data_set_path = _create_data_set(
            num_vectors,
            dimension,
            DEFAULT_TYPE,
            Context.QUERY,
            self.data_set_dir
        )

        # Create a QueryVectorsFromDataSetParamSource with relevant params
        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": DEFAULT_TYPE,
            "data_set_path": data_set_path,
            "k": k,
        }
        query_param_source = QueryVectorsFromDataSetParamSource(
            None, test_param_source_params
        )

        # Check each
        for i in range(num_vectors):
            self._check_params(
                query_param_source.params(),
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                dimension,
                k
            )

        # Assert last call creates stop iteration
        self.assertRaises(
            StopIteration,
            lambda: query_param_source.params()
        )

    def _check_params(
            self,
            params: dict,
            expected_index: str,
            expected_field: str,
            expected_dimension: int,
            expected_k: int
    ):
        index_name = params.get("index")
        self.assertEqual(expected_index, index_name)
        body = params.get("body")
        self.assertIsInstance(body, dict)
        query = body.get("query")
        self.assertIsInstance(query, dict)
        query_knn = query.get("knn")
        self.assertIsInstance(query_knn, dict)
        field = query_knn.get(expected_field)
        self.assertIsInstance(field, dict)
        vector = field.get("vector")
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), expected_dimension)
        k = field.get("k")
        self.assertEqual(k, expected_k)


class BulkVectorsFromDataSetParamSourceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.data_set_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.data_set_dir)

    def test_params(self):
        num_vectors = 49
        bulk_size = 10
        dimension = 10
        data_set_path = _create_data_set(
            num_vectors,
            dimension,
            DEFAULT_TYPE,
            Context.INDEX,
            self.data_set_dir
        )

        test_param_source_params = {
            "index": DEFAULT_INDEX_NAME,
            "field": DEFAULT_FIELD_NAME,
            "data_set_format": DEFAULT_TYPE,
            "data_set_path": data_set_path,
            "bulk_size": bulk_size
        }
        bulk_param_source = BulkVectorsFromDataSetParamSource(
            None, test_param_source_params
        )

        # Check each payload returned
        vectors_consumed = 0
        while vectors_consumed < num_vectors:
            expected_num_vectors = min(num_vectors - vectors_consumed, bulk_size)
            self._check_params(
                bulk_param_source.params(),
                DEFAULT_INDEX_NAME,
                DEFAULT_FIELD_NAME,
                dimension,
                expected_num_vectors
            )
            vectors_consumed += expected_num_vectors

        # Assert last call creates stop iteration
        self.assertRaises(
            StopIteration,
            lambda: bulk_param_source.params()
        )

    def _check_params(
            self,
            params: dict,
            expected_index: str,
            expected_field: str,
            expected_dimension: int,
            expected_num_vectors_in_payload: int
    ):
        size = params.get("size")
        self.assertEqual(size, expected_num_vectors_in_payload)
        body = params.get("body")
        self.assertIsInstance(body, list)
        self.assertEqual(len(body) // 2, expected_num_vectors_in_payload)

        # Bulk payload has 2 parts: first one is the header and the second one
        # is the body. The header will have the index name and the body will
        # have the vector
        for header, req_body in zip(*[iter(body)] * 2):
            index = header.get("index")
            self.assertIsInstance(index, dict)
            index_name = index.get("_index")
            self.assertEqual(index_name, expected_index)

            vector = req_body.get(expected_field)
            self.assertIsInstance(vector, list)
            self.assertEqual(len(vector), expected_dimension)


def _create_data_set(
        num_vectors: int,
        dimension: int,
        extension: str,
        data_set_context: Context,
        data_set_dir
) -> str:
    file_name_base = ''.join(random.choice(string.ascii_letters) for _ in
                             range(8))
    data_set_file_name = "{}.{}".format(file_name_base, extension)
    data_set_path = os.path.join(data_set_dir, data_set_file_name)
    context = DataSetBuildContext(
        data_set_context,
        create_random_2d_array(num_vectors, dimension),
        data_set_path)

    if extension == HDF5DataSet.FORMAT_NAME:
        HDF5Builder().add_data_set_build_context(context).build()
    else:
        BigANNBuilder().add_data_set_build_context(context).build()

    return data_set_path


if __name__ == '__main__':
    unittest.main()
