# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.


def register(registry):
    registry.register_runner("custom-vector-bulk",
                             BulkVectorsFromDataSetRunner(), async_runner=True)
    registry.register_runner("custom-refresh", CustomRefreshRunner(),
                             async_runner=True)


class BulkVectorsFromDataSetRunner:

    async def __call__(self, opensearch, params):
        attempts = 10  # TODO: parametrize this
        for _ in range(attempts):
            try:
                return await opensearch.bulk(
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
                return await opensearch.indices.refresh(index=params["index"])
            except:
                pass

    def __repr__(self, *args, **kwargs):
        return "custom-refresh"
