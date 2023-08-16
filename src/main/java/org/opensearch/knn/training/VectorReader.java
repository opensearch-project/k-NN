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

package org.opensearch.knn.training;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.search.SearchRequestBuilder;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchScrollRequestBuilder;
import org.opensearch.client.Client;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.ValidationException;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.knn.index.IndexUtil;
import org.opensearch.search.SearchHit;
import org.opensearch.search.sort.SortOrder;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class VectorReader {

    public static Logger logger = LogManager.getLogger(VectorReader.class);

    private final Client client;
    private final TimeValue scrollTime = new TimeValue(60000);

    /**
     * Constructor
     *
     * @param client used to make search requests against the cluster
     */
    public VectorReader(Client client) {
        this.client = client;
    }

    /**
     * Read vectors from a provided index/field and pass them to vectorConsumer that will do something with them.
     *
     * @param clusterService cluster service to get information about the index
     * @param indexName name of index containing vectors
     * @param fieldName name of field containing vectors
     * @param maxVectorCount maximum number of vectors to return
     * @param searchSize maximum number of vectors to return in a given search
     * @param vectorConsumer consumer used to do something with the collected vectors after each search
     * @param listener ActionListener that should be called once all search operations complete
     */
    public void read(
        ClusterService clusterService,
        String indexName,
        String fieldName,
        int maxVectorCount,
        int searchSize,
        Consumer<List<Float[]>> vectorConsumer,
        ActionListener<SearchResponse> listener
    ) {

        ValidationException validationException = null;

        // Validate arguments
        if (maxVectorCount <= 0) {
            validationException = new ValidationException();
            validationException.addValidationError("maxVectorCount must be >= 0");
        }

        if (searchSize > 10000 || searchSize <= 0) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError("searchSize must be > 0 and <= 10000");
        }

        IndexMetadata indexMetadata = clusterService.state().metadata().index(indexName);
        if (indexMetadata == null) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationError("index \"" + indexName + "\" does not exist");
            throw validationException;
        }

        ValidationException fieldValidationException = IndexUtil.validateKnnField(indexMetadata, fieldName, -1, null);
        if (fieldValidationException != null) {
            validationException = validationException == null ? new ValidationException() : validationException;
            validationException.addValidationErrors(validationException.validationErrors());
        }

        if (validationException != null) {
            throw validationException;
        }

        // Start reading vectors from index
        SearchScrollRequestBuilder searchScrollRequestBuilder = createSearchScrollRequestBuilder();

        ActionListener<SearchResponse> vectorReaderListener = new VectorReaderListener(
            client,
            fieldName,
            maxVectorCount,
            0,
            listener,
            vectorConsumer,
            searchScrollRequestBuilder
        );

        createSearchRequestBuilder(indexName, fieldName, Integer.min(maxVectorCount, searchSize)).execute(vectorReaderListener);
    }

    private SearchRequestBuilder createSearchRequestBuilder(String indexName, String fieldName, int resultSize) {
        ExistsQueryBuilder queryBuilder = new ExistsQueryBuilder(fieldName);

        SearchRequestBuilder searchRequestBuilder = client.prepareSearch(indexName);
        searchRequestBuilder.setScroll(scrollTime);
        searchRequestBuilder.setQuery(queryBuilder);
        searchRequestBuilder.setSize(resultSize);
        searchRequestBuilder.addSort("_doc", SortOrder.ASC);

        // We are only interested in reading vectors from a particular field
        searchRequestBuilder.setFetchSource(fieldName, null);

        return searchRequestBuilder;
    }

    private SearchScrollRequestBuilder createSearchScrollRequestBuilder() {
        SearchScrollRequestBuilder searchScrollRequestBuilder = client.prepareSearchScroll(null);
        searchScrollRequestBuilder.setScroll(scrollTime);
        return searchScrollRequestBuilder;
    }

    private static class VectorReaderListener implements ActionListener<SearchResponse> {

        final Client client;
        final String fieldName;
        final int maxVectorCount;
        int collectedVectorCount;
        final ActionListener<SearchResponse> listener;
        final Consumer<List<Float[]>> vectorConsumer;
        SearchScrollRequestBuilder searchScrollRequestBuilder;

        /**
         * Constructor
         *
         * @param fieldName name of field to read vectors from
         * @param maxVectorCount maximum total number of vectors that should be read from scroll queries
         * @param collectedVectorCount number of vectors that have already been collected
         * @param listener Search Response listener to be called when all queries complete
         * @param vectorConsumer Consumer used to do something with the vectors
         * @param searchScrollRequestBuilder Search scroll request builder used to get next set of vectors
         */
        public VectorReaderListener(
            Client client,
            String fieldName,
            int maxVectorCount,
            int collectedVectorCount,
            ActionListener<SearchResponse> listener,
            Consumer<List<Float[]>> vectorConsumer,
            SearchScrollRequestBuilder searchScrollRequestBuilder
        ) {
            this.client = client;
            this.fieldName = fieldName;
            this.maxVectorCount = maxVectorCount;
            this.collectedVectorCount = collectedVectorCount;
            this.listener = listener;
            this.vectorConsumer = vectorConsumer;
            this.searchScrollRequestBuilder = searchScrollRequestBuilder;
        }

        @Override
        @SuppressWarnings("unchecked")
        public void onResponse(SearchResponse searchResponse) {
            // Get the vectors from the search response
            // Either add the entire set of returned hits, or maxVectorCount - collectedVectorCount hits
            SearchHit[] hits = searchResponse.getHits().getHits();
            int vectorsToAdd = Integer.min(maxVectorCount - collectedVectorCount, hits.length);
            List<Float[]> trainingData = new ArrayList<>();

            for (int i = 0; i < vectorsToAdd; i++) {
                trainingData.add(
                    ((List<Number>) hits[i].getSourceAsMap().get(fieldName)).stream().map(Number::floatValue).toArray(Float[]::new)
                );
            }

            this.collectedVectorCount += trainingData.size();

            // Do something with the vectors
            vectorConsumer.accept(trainingData);

            if (vectorsToAdd <= 0 || this.collectedVectorCount >= maxVectorCount) {
                // Clear scroll context
                String scrollId = searchResponse.getScrollId();

                if (scrollId != null) {
                    client.prepareClearScroll()
                        .addScrollId(scrollId)
                        .execute(ActionListener.wrap(clearScrollResponse -> listener.onResponse(searchResponse), listener::onFailure));
                } else {
                    listener.onResponse(searchResponse);
                }

            } else {
                // Create a new search that starts where the last search left off
                searchScrollRequestBuilder.setScrollId(searchResponse.getScrollId());
                searchScrollRequestBuilder.execute(this);
            }
        }

        @Override
        public void onFailure(Exception e) {
            // Clear scroll context
            String scrollId = searchScrollRequestBuilder.request().scrollId();

            if (scrollId != null) {
                client.prepareClearScroll()
                    .addScrollId(scrollId)
                    .execute(ActionListener.wrap(clearScrollResponse -> listener.onFailure(e), listener::onFailure));
            } else {
                listener.onFailure(e);
            }
        }
    }
}
