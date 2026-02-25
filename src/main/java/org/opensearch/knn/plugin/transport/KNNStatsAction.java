/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * KNNStatsAction class
 */
public class KNNStatsAction extends ActionType<KNNStatsResponse> {

    public static final KNNStatsAction INSTANCE = new KNNStatsAction();
    public static final String NAME = "cluster:admin/knn_stats_action";

    /**
     * Constructor
     */
    private KNNStatsAction() {
        super(NAME, KNNStatsResponse::new);
    }

    @Override
    public Writeable.Reader<KNNStatsResponse> getResponseReader() {
        return KNNStatsResponse::new;
    }
}
