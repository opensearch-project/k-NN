/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

public class KNNProfileAction extends ActionType<KNNProfileResponse> {

    public static final KNNProfileAction INSTANCE = new KNNProfileAction();
    public static final String NAME = "cluster:admin/knn_profile_action";

    /**
     * Constructor
     */
    private KNNProfileAction() {
        super(NAME, KNNProfileResponse::new);
    }

    @Override
    public Writeable.Reader<KNNProfileResponse> getResponseReader() {
        return KNNProfileResponse::new;
    }
}
