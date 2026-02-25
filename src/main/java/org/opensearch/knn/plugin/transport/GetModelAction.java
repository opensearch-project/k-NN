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
 * GetModelAction class
 */
public class GetModelAction extends ActionType<GetModelResponse> {

    public static final GetModelAction INSTANCE = new GetModelAction();
    public static final String NAME = "cluster:admin/knn_get_model_action";

    /**
     * Constructor
     */
    private GetModelAction() {
        super(NAME, GetModelResponse::new);
    }

    @Override
    public Writeable.Reader<GetModelResponse> getResponseReader() {
        return GetModelResponse::new;
    }
}
