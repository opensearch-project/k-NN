# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import json
from io import BytesIO

from opensearchpy.exceptions import ConnectionTimeout
from osbenchmark.worker_coordinator.runner import Query, parse
from .util import parse_int_parameter, parse_string_parameter, calculate_recall
import logging
import time


def register(registry):
    registry.register_runner(
        "custom-vector-bulk", BulkVectorsFromDataSetRunner(), async_runner=True
    )
    registry.register_runner(
        "query-with-recall", QueryWithRecallRunner(), async_runner=True
    )
    registry.register_runner(
        "custom-refresh", CustomRefreshRunner(), async_runner=True
    )
    registry.register_runner(
        "train-model", TrainModelRunner(), async_runner=True
    )
    registry.register_runner(
        "delete-model", DeleteModelRunner(), async_runner=True
    )


class BulkVectorsFromDataSetRunner:

    async def __call__(self, opensearch, params):
        size = parse_int_parameter("size", params)
        retries = parse_int_parameter("retries", params, 0) + 1

        for _ in range(retries):
            try:
                await opensearch.bulk(
                    body=params["body"],
                    timeout='5m'
                )

                return size, "docs"
            except ConnectionTimeout:
                logging.getLogger(__name__)\
                    .warning("Bulk vector ingestion timed out. Retrying")

        raise TimeoutError("Failed to submit bulk request in specified number "
                           "of retries: {}".format(retries))

    def __repr__(self, *args, **kwargs):
        return "custom-vector-bulk"


class QueryWithRecallRunner(Query):

    async def __call__(self, opensearch, params):
        # A query will come with a set of k ground truth nearest neighbors. In
        # addition to latency and the other metrics, we will also give a recall
        # score which will be the number of nearest neighbors returned
        index_name = parse_string_parameter("index", params)
        request_params = params.get("request-params", {})

        # Code adopted from OpenSearch benchmarks implementation of the query:
        # https://github.com/opensearch-project/opensearch-benchmark/blob/0.0.2/osbenchmark/worker_coordinator/runner.py#L757.
        # The reason we have to create a custom runner is that we want the
        # ability to add the recall computation metric.
        response_dict = await self._raw_search(opensearch, None, index_name,
                                               params["body"], request_params)
        response_string = json.dumps(response_dict)
        response_bytes = BytesIO(bytes(response_string, 'utf-8'))
        props = parse(response_bytes,
                      ["hits.total", "hits.total.value",
                       "hits.total.relation", "timed_out", "took"])
        hits_total = props.get("hits.total.value", props.get("hits.total", 0))
        hits_relation = props.get("hits.total.relation", "eq")
        timed_out = props.get("timed_out", False)
        took = props.get("took", 0)

        results = {
            "weight": 1,
            "unit": "ops",
            "success": True,
            "hits": hits_total,
            "hits_relation": hits_relation,
            "timed_out": timed_out,
            "took": took,
        }

        ground_truth = params["ground_truth"]
        if ground_truth is not None:
            # TODO: Debug why this isnt in the output results
            hits = [int(hit["_id"]) for hit in response_dict["hits"]["hits"]]
            results["recall"] = calculate_recall(hits, ground_truth)

        return results

    def __repr__(self, *args, **kwargs):
        return "query-with-recall"


class CustomRefreshRunner:

    async def __call__(self, opensearch, params):
        retries = parse_int_parameter("retries", params, 0) + 1

        for _ in range(retries):
            try:
                await opensearch.indices.refresh(
                    index=parse_string_parameter("index", params)
                )

                return
            except ConnectionTimeout:
                logging.getLogger(__name__)\
                    .warning("Custom refresh timed out. Retrying")

        raise TimeoutError("Failed to refresh the index in specified number "
                           "of retries: {}".format(retries))

    def __repr__(self, *args, **kwargs):
        return "custom-refresh"


class TrainModelRunner:

    async def __call__(self, opensearch, params):
        # Train a model and wait for it training to complete
        body = params["body"]
        timeout = parse_int_parameter("timeout", params)
        model_id = parse_string_parameter("model_id", params)

        method = "POST"
        model_uri = "/_plugins/_knn/models/{}".format(model_id)
        await opensearch.transport.perform_request(method, "{}/_train".format(model_uri), body=body)

        start_time = time.time()
        while time.time() < start_time + timeout:
            time.sleep(1)
            model_response = await opensearch.transport.perform_request("GET", model_uri)

            if 'state' not in model_response.keys():
                continue

            if model_response['state'] == 'created':
                #TODO: Return model size as well
                return 1, "models_trained"

            if model_response['state'] == 'failed':
                raise Exception("Failed to create model: {}".format(model_response))

        raise Exception('Failed to create model: {} within timeout {} seconds'
                        .format(model_id, timeout))

    def __repr__(self, *args, **kwargs):
        return "train-model"


class DeleteModelRunner:

    async def __call__(self, opensearch, params):
        # Delete model provided by model id
        method = "DELETE"
        model_id = parse_string_parameter("model_id", params)
        uri = "/_plugins/_knn/models/{}".format(model_id)

        # Ignore if model doesnt exist
        await opensearch.transport.perform_request(method, uri, params={"ignore": [400, 404]})

    def __repr__(self, *args, **kwargs):
        return "delete-model"
