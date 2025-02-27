/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * Action used to detect if the KNNCircuit has been tripped cluster wide
 */
public class KNNCircuitBreakerTrippedAction extends ActionType<KNNCircuitBreakerTrippedResponse> {

    public static final String NAME = "cluster:admin/knn_circuit_breaker_tripped_action";
    public static final KNNCircuitBreakerTrippedAction INSTANCE = new KNNCircuitBreakerTrippedAction(
        NAME,
        KNNCircuitBreakerTrippedResponse::new
    );

    /**
     * Constructor
     *
     * @param name name of action
     * @param responseReader reader for the KNNCircuitBreakerTrippedResponse
     */
    public KNNCircuitBreakerTrippedAction(String name, Writeable.Reader<KNNCircuitBreakerTrippedResponse> responseReader) {
        super(name, responseReader);
    }
}
