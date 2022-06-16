/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.common.io.stream.Writeable;

/**
 * Action to update blocked modelIds list
 */
public class UpdateBlockedModelAction extends ActionType<AcknowledgedResponse> {

    public static final String NAME = "cluster:admin/knn_update_blocked_model_action";
    public static final UpdateBlockedModelAction INSTANCE = new UpdateBlockedModelAction(NAME, AcknowledgedResponse::new);

    /**
     * Constructor.
     *
     * @param name name of action
     * @param acknowledgedResponseReader reader for acknowledged response
     */
    public UpdateBlockedModelAction(String name, Writeable.Reader<AcknowledgedResponse> acknowledgedResponseReader) {
        super(name, acknowledgedResponseReader);
    }
}
