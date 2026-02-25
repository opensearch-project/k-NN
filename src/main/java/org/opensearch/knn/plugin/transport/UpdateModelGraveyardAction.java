/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.ActionType;
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * Action to update model graveyard
 */
public class UpdateModelGraveyardAction extends ActionType<AcknowledgedResponse> {

    public static final String NAME = "cluster:admin/knn_update_model_graveyard_action";
    public static final UpdateModelGraveyardAction INSTANCE = new UpdateModelGraveyardAction(NAME, AcknowledgedResponse::new);

    /**
     * Constructor.
     *
     * @param name name of action
     * @param acknowledgedResponseReader reader for acknowledged response
     */
    public UpdateModelGraveyardAction(String name, Writeable.Reader<AcknowledgedResponse> acknowledgedResponseReader) {
        super(name, acknowledgedResponseReader);
    }
}
