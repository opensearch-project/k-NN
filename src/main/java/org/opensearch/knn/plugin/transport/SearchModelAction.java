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
import org.opensearch.action.search.SearchResponse;
import org.opensearch.core.common.io.stream.Writeable;

/**
 * GetModelAction class
 */
public class SearchModelAction extends ActionType<SearchResponse> {

    public static final SearchModelAction INSTANCE = new SearchModelAction();
    public static final String NAME = "cluster:admin/knn_search_model_action";

    /**
     * Constructor
     */
    private SearchModelAction() {
        super(NAME, SearchResponse::new);
    }

    @Override
    public Writeable.Reader<SearchResponse> getResponseReader() {
        return SearchResponse::new;
    }
}
