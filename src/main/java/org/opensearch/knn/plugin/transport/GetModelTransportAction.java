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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

/**
 * Transport Action for {@link GetModelAction}
 */
public class GetModelTransportAction extends HandledTransportAction<GetModelRequest, GetModelResponse> {
    private static final Logger LOG = LogManager.getLogger(GetModelTransportAction.class);
    private ModelDao modelDao;

    @Inject
    public GetModelTransportAction(TransportService transportService, ActionFilters actionFilters) {
        super(GetModelAction.NAME, transportService, actionFilters, GetModelRequest::new);
        this.modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
    }

    @Override
    protected void doExecute(Task task, GetModelRequest request, ActionListener<GetModelResponse> actionListener) {
        String modelID = request.getModelID();

        modelDao.get(modelID, actionListener);

    }
}
