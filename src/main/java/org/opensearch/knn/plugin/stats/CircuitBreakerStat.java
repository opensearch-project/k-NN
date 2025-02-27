/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.Version;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedAction;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedRequest;
import org.opensearch.transport.client.Client;

import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.opensearch.knn.index.KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED;

public class CircuitBreakerStat extends KNNStat<Boolean> {

    public static final String CONTEXT_CB_TRIPPED = "is_cb_tripped";

    private static final Function<KNNStatFetchContext, Boolean> FETCHER = context -> {
        if (context == null) {
            return false;
        }
        return (Boolean) context.getContext(StatNames.CIRCUIT_BREAKER_TRIGGERED.getName()).get(CONTEXT_CB_TRIPPED);
    };

    private final Client client;
    private final Supplier<Version> minVersionSupplier;

    public CircuitBreakerStat(Client client, Supplier<Version> minVersionSupplier) {
        super(true, FETCHER);
        this.client = client;
        this.minVersionSupplier = minVersionSupplier;
    }

    @Override
    public ActionListener<Void> setupContext(KNNStatFetchContext knnStatFetchContext, ActionListener<Void> actionListener) {
        // If there are any nodes in the cluster before 3.0, then we need to fall back to checking the CB
        if (minVersionSupplier.get().compareTo(Version.V_3_0_0) < 0) {
            return ActionListener.wrap(knnCircuitBreakerTrippedResponse -> {
                knnStatFetchContext.addContext(
                    StatNames.CIRCUIT_BREAKER_TRIGGERED.getName(),
                    Map.of(CONTEXT_CB_TRIPPED, KNNSettings.state().getSettingValue(KNN_CIRCUIT_BREAKER_TRIGGERED))
                );
                actionListener.onResponse(null);
            }, actionListener::onFailure);
        }
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
