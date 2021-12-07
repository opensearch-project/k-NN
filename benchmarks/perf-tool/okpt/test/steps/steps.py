# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Provides steps for OpenSearch tests.

Some of the OpenSearch operations return a `took` field in the response body,
so the profiling decorators aren't needed for some functions.
"""
import json
from typing import Any, Dict, List, cast

import numpy as np
import requests
import time

from opensearchpy import OpenSearch, RequestsHttpConnection

from okpt.io.config.parsers.base import ConfigurationError
from okpt.io.config.parsers.util import parse_string_param, parse_int_param, parse_dataset, parse_bool_param
from okpt.io.utils.reader import parse_json_from_path
from okpt.test.steps import base
from okpt.test.steps.base import StepConfig


class OpenSearchStep(base.Step):
    """See base class."""

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.endpoint = parse_string_param("endpoint", step_config.config, step_config.implicit_config, "localhost")
        default_port = 9200 if self.endpoint == "localhost" else 80
        self.port = parse_int_param("port", step_config.config, step_config.implicit_config, default_port)
        self.opensearch = get_opensearch_client(str(self.endpoint), int(self.port))


class CreateIndexStep(OpenSearchStep):
    """See base class."""

    label = 'create_index'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.index_name = parse_string_param("index_name", step_config.config, {}, None)
        index_spec = parse_string_param("index_spec", step_config.config, {}, None)
        self.body = parse_json_from_path(index_spec)
        if self.body is None:
            raise ConfigurationError("Index body must be passed in")

    def _action(self):
        """Creates an OpenSearch index, applying the index settings/mappings.

        Returns:
            An OpenSearch index creation response body.
        """
        self.opensearch.indices.create(index=self.index_name, body=self.body)
        return dict()

    def _get_measures(self) -> List[str]:
        return ['took']


class DisableRefreshStep(OpenSearchStep):
    """See base class."""

    label = 'disable_refresh'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)

    def _action(self):
        """Disables the refresh interval for an OpenSearch index.

        Returns:
            An OpenSearch index settings update response body.
        """
        self.opensearch.indices.put_settings(
            body={'index': {
                'refresh_interval': -1
            }})

        return dict()

    def _get_measures(self) -> List[str]:
        return ['took']


class RefreshIndexStep(OpenSearchStep):
    """See base class."""

    label = 'refresh_index'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.index_name = parse_string_param("index_name", step_config.config, {}, None)

    def _action(self):
        while True:
            try:
                self.opensearch.indices.refresh(index=self.index_name)
                return dict()
            except:
                pass

    def _get_measures(self) -> List[str]:
        return ['took']


class ForceMergeStep(OpenSearchStep):
    """See base class."""

    label = 'force_merge'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.index_name = parse_string_param("index_name", step_config.config, {}, None)
        self.max_num_segments = parse_int_param("max_num_segments", step_config.config, {}, None)

    def _action(self):
        while True:
            try:
                self.opensearch.indices.forcemerge(index=self.index_name, max_num_segments=self.max_num_segments)
                return dict()
            except:
                pass

    def _get_measures(self) -> List[str]:
        return ['took']


class TrainModelStep(OpenSearchStep):
    label = 'train_model'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)

        self.model_id = parse_string_param("model_id", step_config.config, {}, "Test")
        self.train_index_name = parse_string_param("train_index", step_config.config, {}, None)
        self.train_index_field = parse_string_param("train_field", step_config.config, {}, None)
        self.dimension = parse_int_param("dimension", step_config.config, {}, None)
        self.description = parse_string_param("description", step_config.config, {}, "Default")
        self.max_training_vector_count = parse_int_param("max_training_vector_count", step_config.config, {}, 10000000000000)

        method_spec = parse_string_param("method_spec", step_config.config, {}, None)
        self.method = parse_json_from_path(method_spec)
        if self.method is None:
            raise ConfigurationError("method must be passed in")

    def _action(self):
        """Train a model for an index.

        Returns:
            The trained model
        """

        # Build body
        body = {
            "training_index": self.train_index_name,
            "training_field": self.train_index_field,
            "description": self.description,
            "dimension": self.dimension,
            "method": self.method,
            "max_training_vector_count": self.max_training_vector_count
        }

        # So, we trained the model. Now we need to wait until we have to wait until the model is created. Poll every
        # 1/10 second
        requests.post(
            "http://" + self.endpoint + ":" + str(self.port) + "/_plugins/_knn/models/" + str(self.model_id) + "/_train",
            json.dumps(body),
            headers={"content-type": "application/json"})

        sleep_time = 0.1
        timeout = 100000
        i = 0
        while i < timeout:
            time.sleep(sleep_time)
            model_response = get_model(self.endpoint, self.port, self.model_id)
            if "state" in model_response.keys() and model_response["state"] == "created":
                return dict()
            i += 1

        raise TimeoutError("Failed to create model")

    def _get_measures(self) -> List[str]:
        return ['took']


class DeleteModelStep(OpenSearchStep):
    label = 'delete_model'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)

        self.model_id = parse_string_param("model_id", step_config.config, {}, "Test")

    def _action(self):
        """Train a model for an index.

        Returns:
            The trained model
        """
        delete_model(self.endpoint, self.port, self.model_id)
        return dict()

    def _get_measures(self) -> List[str]:
        return ['took']


class DeleteIndexStep(OpenSearchStep):
    label = 'delete_index'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)

        self.index_name = parse_string_param("index_name", step_config.config, {}, None)

    def _action(self):
        """Delete the index

        Returns:
            An empty dict
        """
        delete_index(self.opensearch, self.index_name)
        return dict()

    def _get_measures(self) -> List[str]:
        return ['took']


class IngestStep(OpenSearchStep):

    label = 'ingest'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.index_name = parse_string_param("index_name", step_config.config, {}, None)
        self.field_name = parse_string_param("field_name", step_config.config, {}, None)
        self.bulk_size = parse_int_param("bulk_size", step_config.config, {}, 300)
        self.implicit_config = step_config.implicit_config
        dataset_format = parse_string_param("dataset_format", step_config.config, {}, "hdf5")
        dataset_path = parse_string_param("dataset_path", step_config.config, {}, None)
        self.dataset = parse_dataset(dataset_path, dataset_format)

    def _action(self):
        results = dict()
        def action(doc_id): return {'index': {'_index': self.index_name, "_id": doc_id}}

        i = 0
        index_responses = list()
        while i < self.dataset.train.len():
            partition = cast(np.ndarray, self.dataset.train[i:i + self.bulk_size])
            body = bulk_transform(partition, self.field_name, action, i)
            result = bulk_index(self.opensearch, self.index_name, body)
            index_responses.append(result)
            i += self.bulk_size

        results["took"] = [float(index_response["took"]) for index_response in index_responses]
        results["store_kb"] = get_index_size_in_kb(self.opensearch, self.index_name)

        return results

    def _get_measures(self) -> List[str]:
        return ['took', 'store_kb']


class QueryStep(OpenSearchStep):

    label = 'query'

    def __init__(self, step_config: StepConfig):
        super().__init__(step_config)
        self.k = parse_int_param("k", step_config.config, {}, 100)
        self.r = parse_int_param("r", step_config.config, {}, 1)
        self.index_name = parse_string_param("index_name", step_config.config, {}, None)
        self.field_name = parse_string_param("field_name", step_config.config, {}, None)
        self.calculate_recall = parse_bool_param("calculate_recall", step_config.config, {}, False)
        dataset_format = parse_string_param("dataset_format", step_config.config, {}, "hdf5")
        dataset_path = parse_string_param("dataset_path", step_config.config, {}, None)
        self.dataset = parse_dataset(dataset_path, dataset_format)
        self.implicit_config = step_config.implicit_config

    def _action(self):
        def get_body(vec): return {
            'size': self.k,
            'query': {
                'knn': {
                    self.field_name: {
                        'vector': vec,
                        'k': self.k
                    }
                }
            }
        }

        results = dict()
        query_responses = list()
        for v in self.dataset.test:
            query_responses.append(query_index(self.opensearch, self.index_name, get_body(v), [self.field_name]))

        results["took"] = [float(query_response["took"]) for query_response in query_responses]
        results["memory_kb"] = get_cache_size_in_kb(self.endpoint, 80)

        if self.calculate_recall:
            ids = [[int(hit["_id"]) for hit in query_response["hits"]["hits"]] for query_response in query_responses]
            results["recall@K"] = recall_at_r(ids,  self.dataset.neighbors, self.k, self.k)
            results[f'recall@{str(self.r)}'] = recall_at_r(ids,  self.dataset.neighbors, self.r, self.k)

        return results

    def _get_measures(self) -> List[str]:
        measures = ['took', 'memory_kb']

        if self.calculate_recall:
            measures.extend(['recall@K', f'recall@{str(self.r)}'])

        return measures


# Helper functions - (AKA not steps)
def bulk_transform(partition: np.ndarray, field_name: str, action, offset: int) -> List[Dict[str, Any]]:
    """Partitions and transforms a list of vectors into OpenSearch's bulk injection format.
    Args:
        offset: to start counting from
        partition: An array of vectors to transform.
        field_name: field name for action
        action: Bulk API action.
    Returns:
        An array of transformed vectors in bulk format.
    """
    actions = list()
    [actions.extend([action(i + offset), None]) for i in range(len(partition))]
    actions[1::2] = [{field_name: vec} for vec in partition.tolist()]
    return actions


def delete_index(opensearch: OpenSearch, index_name: str):
    """Deletes an OpenSearch index.

    Args:
        opensearch: An OpenSearch client.
        index_name: Name of the OpenSearch index to be deleted.
    """
    opensearch.indices.delete(index=index_name, ignore=[400, 404])


def get_model(endpoint, port, model_id):
    """
    Retrieve a model from an OpenSearch cluster
    Args:
        endpoint: Endpoint OpenSearch is running on
        port: Port OpenSearch is running on
        model_id: ID of model to be deleted
    Returns:
        Get model response
    """
    response = requests.get(
        "http://" + endpoint + ":" + str(port) + "/_plugins/_knn/models/" + model_id,
        headers={"content-type": "application/json"})
    return response.json()


def delete_model(endpoint, port, model_id):
    """
    Deletes a model from OpenSearch cluster
    Args:
        endpoint: Endpoint OpenSearch is running on
        port: Port OpenSearch is running on
        model_id: ID of model to be deleted
    Returns:
        Deleted model response
    """
    response = requests.delete(
        "http://" + endpoint + ":" + str(port) + "/_plugins/_knn/models/" + model_id,
        headers={"content-type": "application/json"})
    return response.json()


def get_opensearch_client(endpoint: str, port: int):
    """
    Get an opensearch client from an endpoint and port
    Args:
        endpoint: Endpoint OpenSearch is running on
        port: Port OpenSearch is running on
    Returns:
        OpenSearch client

    """
    # TODO: fix for security in the future
    return OpenSearch(
        hosts=[{
            'host': endpoint,
            'port': port
        }],
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


def recall_at_r(results, ground_truth_set, r, k):
    """
    Calculates the recall@R for a set of queries against a ground truth nearest neighbor set
    Args:
        results: 2D list containing ids of results returned by OpenSearch. results[i][j] i refers to query, j refers to
            result in the query
        ground_truth_set: 2D list containing ids of the true nearest neighbors for a set of queries
        r: number of top results to check if they are in the ground truth k-NN set.
        k: k value for the query
    Returns:
        Recall at R
    """
    correct = 0.0
    for i, true_neighbors in enumerate(ground_truth_set):
        true_neighbors_set = set(true_neighbors[:k])
        for j in range(r):
            if results[i][j] in true_neighbors_set:
                correct += 1.0

    return correct / (r * len(ground_truth_set))


def get_index_size_in_kb(opensearch, index_name):
    """
    Gets the size of an index in kilobytes
    Args:
        opensearch: opensearch client
        index_name: name of index to look up
    Returns:
        size of index in kilobytes
    """
    return int(opensearch.indices.stats(index_name, metric="store")["indices"][index_name]["total"]["store"]
               ["size_in_bytes"]) / 1024


def get_cache_size_in_kb(endpoint, port):
    """
    Gets the size of the k-NN cache in kilobytes
    Args:
        endpoint: endpoint of OpenSearch cluster
        port: port of endpoint OpenSearch is running on
    Returns:
        size of cache in kilobytes
    """
    response = requests.get(
        "http://" + endpoint + ":" + str(port) + "/_plugins/_knn/stats",
        headers={"content-type": "application/json"})
    stats = response.json()

    keys = stats["nodes"].keys()

    total_used = 0
    for key in keys:
        total_used += int(stats["nodes"][key]["graph_memory_usage"])
    return total_used


def query_index(opensearch: OpenSearch, index_name: str, body: dict, excluded_fields: list):
    return opensearch.search(index=index_name, body=body, _source_excludes=excluded_fields)


def bulk_index(opensearch: OpenSearch, index_name: str, body: dict):
    return opensearch.bulk(index=index_name, body=body, timeout="5m")
