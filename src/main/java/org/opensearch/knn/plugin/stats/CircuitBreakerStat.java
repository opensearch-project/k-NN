/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedAction;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedRequest;
import org.opensearch.transport.client.Client;

import java.util.Map;
import java.util.function.Function;

public class CircuitBreakerStat extends KNNStat<Boolean> {

    public static final String CONTEXT_CB_TRIPPED = "is_cb_tripped";

    private static final Function<KNNStatFetchContext, Boolean> FETCHER = context -> {
        if (context == null) {
            return false;
        }
        return (Boolean) context.getContext(StatNames.CIRCUIT_BREAKER_TRIGGERED.getName()).get(CONTEXT_CB_TRIPPED);
    };

    private final Client client;

    public CircuitBreakerStat(Client client) {
        super(true, FETCHER);
        this.client = client;
    }

    @Override
    public ActionListener<Void> setupContext(KNNStatFetchContext knnStatFetchContext, ActionListener<Void> actionListener) {
        return ActionListener.wrap(
            response -> client.execute(
                KNNCircuitBreakerTrippedAction.INSTANCE,
                new KNNCircuitBreakerTrippedRequest(),
                ActionListener.wrap(knnCircuitBreakerTrippedResponse -> {
                    knnStatFetchContext.addContext(
                        StatNames.CIRCUIT_BREAKER_TRIGGERED.getName(),
                        Map.of(CONTEXT_CB_TRIPPED, knnCircuitBreakerTrippedResponse.isTripped())
                    );
                    actionListener.onResponse(null);
                }, actionListener::onFailure)
            ),
            actionListener::onFailure
        );
    }
}
