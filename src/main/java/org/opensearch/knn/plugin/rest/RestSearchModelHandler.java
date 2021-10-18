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

package org.opensearch.knn.plugin.rest;

import com.google.common.collect.ImmutableList;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.SearchModelAction;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestResponse;
import org.opensearch.rest.RestStatus;
import org.opensearch.rest.action.RestCancellableNodeClient;
import org.opensearch.rest.action.RestResponseListener;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.rest.action.search.RestSearchAction;
import org.opensearch.search.SearchHit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.IntConsumer;

import static org.opensearch.common.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;

/**
 * Rest Handler for search model api endpoint.
 */
public class RestSearchModelHandler extends BaseRestHandler {

    private final static String NAME = "knn_search_model_action";
    private static final String SEARCH = "_search";
    //Add params that are not fit to be part of model search
    public List<String> UNSUPPORTED_PARAM_LIST = Arrays.asList(
        "index" //we don't want to search across all indices
    );

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.GET,
                    String.format(Locale.ROOT, "%s/%s/%s", KNNPlugin.KNN_BASE_URI, MODELS, SEARCH)
                ),
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/%s/%s", KNNPlugin.KNN_BASE_URI, MODELS, SEARCH)
                )
            );
    }

    private void checkUnSupportedParamsExists(RestRequest request) {
        List<String> invalidParam = new ArrayList<>();

        UNSUPPORTED_PARAM_LIST.forEach(param -> {
            if (request.hasParam(param)) {
                invalidParam.add(param);
            }
        });
        if (invalidParam.isEmpty())
            return;
        String errorMessage = "request contains an unrecognized parameter: [ " + String.join(",", invalidParam) + " ]";
        throw new IllegalArgumentException(errorMessage);
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        checkUnSupportedParamsExists(request);
        SearchRequest searchRequest = new SearchRequest();
        IntConsumer setSize = size -> searchRequest.source().size(size);
        request.withContentOrSourceParamParserOrNull(parser ->
            RestSearchAction.parseSearchRequest(searchRequest, request, parser, client.getNamedWriteableRegistry(), setSize));

        return channel -> {
            RestCancellableNodeClient cancelClient = new RestCancellableNodeClient(client, request.getHttpChannel());
            cancelClient.execute(SearchModelAction.INSTANCE, searchRequest, new RestToXContentListener<>(channel));
        };
    }
}
