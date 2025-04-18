/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;

/**
 * Action for profiling KNN vectors in an index
 */
public class KNNProfileAction extends ActionType<KNNProfileResponse> {
    public static final String NAME = "indices:knn/vector/profile";
    public static final KNNProfileAction INSTANCE = new KNNProfileAction();

    private KNNProfileAction() {
        super(NAME, KNNProfileResponse::new);
    }
}
