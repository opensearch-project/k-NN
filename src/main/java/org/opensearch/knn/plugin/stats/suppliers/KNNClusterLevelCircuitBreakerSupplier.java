/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats.suppliers;

import lombok.AllArgsConstructor;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedAction;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedRequest;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedResponse;
import org.opensearch.transport.client.Client;

import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;

@AllArgsConstructor
public class KNNClusterLevelCircuitBreakerSupplier implements Supplier<Boolean> {

    private final Client client;

    @Override
    public Boolean get() {
        try {
            KNNCircuitBreakerTrippedResponse knnCircuitBreakerTrippedResponse = client.execute(
                KNNCircuitBreakerTrippedAction.INSTANCE,
                new KNNCircuitBreakerTrippedRequest()
            ).get();
            return knnCircuitBreakerTrippedResponse.isTripped();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
