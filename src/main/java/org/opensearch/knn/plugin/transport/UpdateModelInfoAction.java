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
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.common.io.stream.Writeable;

/**
 * Action to update model info.
 */
public class UpdateModelInfoAction extends ActionType<AcknowledgedResponse> {

    public static final String NAME = "cluster:admin/knn_update_model_info_action";
    public static final UpdateModelInfoAction INSTANCE = new UpdateModelInfoAction(NAME, AcknowledgedResponse::new);

    /**
     * Constructor.
     *
     * @param name name of action
     * @param acknowledgedResponseReader reader for acknowledged response
     */
    public UpdateModelInfoAction(String name, Writeable.Reader<AcknowledgedResponse> acknowledgedResponseReader) {
        super(name, acknowledgedResponseReader);
    }
}
