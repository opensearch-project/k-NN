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

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

@Log4j2
public class DeleteModelTransportAction extends HandledTransportAction<DeleteModelRequest, DeleteModelResponse> {

    private final ModelDao modelDao;

    @Inject
    public DeleteModelTransportAction(TransportService transportService, ActionFilters filters) {
        super(DeleteModelAction.NAME, transportService, filters, DeleteModelRequest::new);
        this.modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
    }

    @Override
    protected void doExecute(Task task, DeleteModelRequest request, ActionListener<DeleteModelResponse> listener) {
        String modelID = request.getModelID();
        modelDao.delete(modelID, ActionListener.wrap(listener::onResponse, e -> {
            log.error(e);
            listener.onFailure(e);
        }));
    }
}
