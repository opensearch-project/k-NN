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
import org.opensearch.action.support.clustermanager.AcknowledgedResponse;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * Action to update model metadata.
 */
public class UpdateModelMetadataAction extends ActionType<AcknowledgedResponse> {

    public static final String NAME = "cluster:admin/knn_update_model_metadata_action";
    public static final UpdateModelMetadataAction INSTANCE = new UpdateModelMetadataAction(NAME, AcknowledgedResponse::new);

    /**
     * Constructor.
     *
     * @param name name of action
     * @param acknowledgedResponseReader reader for acknowledged response
     */
    public UpdateModelMetadataAction(String name, Writeable.Reader<AcknowledgedResponse> acknowledgedResponseReader) {
        super(name, acknowledgedResponseReader);
    }
}
