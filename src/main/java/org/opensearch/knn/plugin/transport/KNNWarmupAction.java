/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

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
