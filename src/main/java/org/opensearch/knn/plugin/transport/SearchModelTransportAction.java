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
import org.opensearch.common.util.concurrent.ThreadContext;
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
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            this.modelDao.search(request, listener);
        } catch (IOException e) {
            logger.error(e);
            listener.onFailure(e);
        }
    }
}
