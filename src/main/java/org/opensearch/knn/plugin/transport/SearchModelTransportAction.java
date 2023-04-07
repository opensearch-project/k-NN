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

import org.opensearch.action.ActionListener;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.common.inject.Inject;
import org.opensearch.knn.common.TaskRunner;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import java.io.IOException;

public class SearchModelTransportAction extends HandledTransportAction<SearchRequest, SearchResponse> {
    private ModelDao modelDao;

    private final Client client;

    @Inject
    public SearchModelTransportAction(TransportService transportService, ActionFilters actionFilters, Client client) {
        super(SearchModelAction.NAME, transportService, actionFilters, SearchRequest::new);
        this.modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
        this.client = client;
    }

    @Override
    protected void doExecute(Task task, SearchRequest request, ActionListener<SearchResponse> listener) {
        TaskRunner.runWithStashedThreadContext(client, () -> {
            try {
                this.modelDao.search(request, listener);
            } catch (IOException e) {
                listener.onFailure(e);
            }
        });
    }
}
