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
import org.opensearch.common.io.stream.Writeable;

/**
 * Action associated with k-NN warmup
 */
public class KNNWarmupAction extends ActionType<KNNWarmupResponse> {

    public static final KNNWarmupAction INSTANCE = new KNNWarmupAction();
    public static final String NAME = "cluster:admin/knn_warmup_action";

    private KNNWarmupAction() {
        super(NAME, KNNWarmupResponse::new);
    }

    @Override
    public Writeable.Reader<KNNWarmupResponse> getResponseReader() {
        return KNNWarmupResponse::new;
    }
}
