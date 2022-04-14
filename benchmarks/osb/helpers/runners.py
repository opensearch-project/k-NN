# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from helpers.util import parse_int_parameter, parse_string_parameter
import time


def register(registry):
    registry.register_runner(
        "custom-vector-bulk", BulkVectorsFromDataSetRunner(), async_runner=True
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
        for _ in range(params["retries"] + 1):
            try:
                return await opensearch.bulk(
                    body=params["body"],
                    timeout='5m'
                )
            except:
                pass

    def __repr__(self, *args, **kwargs):
        return "custom-vector-bulk"


class CustomRefreshRunner:

    async def __call__(self, opensearch, params):
        # Basically just keep calling it until it succeeds
        attempts = parse_int_parameter("retries", params, 0) + 1

        for _ in range(attempts):
            try:
                return await opensearch.indices.refresh(
                    index=parse_string_parameter("index", params)
                )
            except:
                pass

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
        i = 0
        while i < timeout:
            time.sleep(1)
            model_response = await opensearch.transport.perform_request("GET", model_uri)

            if 'state' not in model_response.keys():
                continue

            if model_response['state'] == 'created':
                return {}

            if model_response['state'] == 'failed':
                raise Error("Failed to create model: {}".format(model_response))

            i += 1

        raise TimeoutError('Failed to create model: {}'.format(model_id))

    def __repr__(self, *args, **kwargs):
        return "train-model"


class DeleteModelRunner:

    async def __call__(self, opensearch, params):
        # Delete model provided by model id
        method = "DELETE"
        model_id = parse_string_parameter("model_id", params)
        uri = "/_plugins/_knn/models/{}".format(model_id)

        # Ignore if model doesnt exist
        return await opensearch.transport.perform_request(method, uri, params={"ignore": [400, 404]})

    def __repr__(self, *args, **kwargs):
        return "delete-model"
