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
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.common.inject.Inject;
import org.opensearch.knn.common.TaskRunner;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

/**
 * Transport Action for {@link GetModelAction}
 */
public class GetModelTransportAction extends HandledTransportAction<GetModelRequest, GetModelResponse> {
    private static final Logger LOG = LogManager.getLogger(GetModelTransportAction.class);
    private ModelDao modelDao;

    private final Client client;

    @Inject
    public GetModelTransportAction(TransportService transportService, ActionFilters actionFilters, Client client) {
        super(GetModelAction.NAME, transportService, actionFilters, GetModelRequest::new);
        this.modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        this.client = client;
    }

    @Override
    protected void doExecute(Task task, GetModelRequest request, ActionListener<GetModelResponse> actionListener) {
        TaskRunner.runWithStashedThreadContext(client, () -> {
            String modelID = request.getModelID();
            modelDao.get(modelID, actionListener);
        });
    }
}
