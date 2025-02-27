/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedAction;
import org.opensearch.knn.plugin.transport.KNNCircuitBreakerTrippedRequest;
import org.opensearch.transport.client.Client;

import java.util.function.Supplier;

import static org.opensearch.knn.index.KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED;

/**
 * Cluster stat that checks if the circuit breaker is enabled. For clusters on or after version 3.0, this stat will
 * be populated by broadcasting a transport call to all nodes to see if any of their circuit breakers are set. Before
 * 3.0, it checks the cluster setting.
 */
public class CircuitBreakerStat extends KNNStat<Boolean> {

    private final Client client;
    private final Supplier<Version> minVersionSupplier;
    @Getter
    private Boolean cbTripped;

    /**
     *
     * @param client Client used to execute transport call to get CB values of nodes
     * @param minVersionSupplier Minimum version supplier to provide minimum node version in the cluster
     */
    public CircuitBreakerStat(Client client, Supplier<Version> minVersionSupplier) {
        super(true, null);
        this.client = client;
        this.minVersionSupplier = minVersionSupplier;
        this.cbTripped = null;
    }

    @Override
    public ActionListener<Void> setupContext(ActionListener<Void> actionListener) {
        // If there are any nodes in the cluster before 3.0, then we need to fall back to checking the CB via cluster
        // setting
        if (minVersionSupplier.get().before(Version.V_3_0_0)) {
            return ActionListener.wrap(voidResponse -> {
                cbTripped = KNNSettings.state().getSettingValue(KNN_CIRCUIT_BREAKER_TRIGGERED);
                actionListener.onResponse(voidResponse);
            }, actionListener::onFailure);
        }
        return ActionListener.wrap(
            voidResponse -> client.execute(
                KNNCircuitBreakerTrippedAction.INSTANCE,
                new KNNCircuitBreakerTrippedRequest(),
                ActionListener.wrap(knnCircuitBreakerTrippedResponse -> {
                    cbTripped = knnCircuitBreakerTrippedResponse.isTripped();
                    actionListener.onResponse(voidResponse);
                }, actionListener::onFailure)
            ),
            actionListener::onFailure
        );
    }

    @Override
    public Boolean getValue() {
        assert cbTripped != null;
        return cbTripped;
    }
}
