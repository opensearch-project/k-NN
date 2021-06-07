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
import org.opensearch.action.ActionListener;
import org.opensearch.action.search.SearchRequestBuilder;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchScrollRequestBuilder;
import org.opensearch.client.Client;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.ExistsQueryBuilder;
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.index.KNNVectorFieldMapper;
import org.opensearch.search.SearchHit;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class VectorReader {


    public static Logger logger = LogManager.getLogger(VectorReader.class);

    private final IndicesService indicesService;
    private final Client client;
    private final TimeValue scrollTime = new TimeValue(60000);

    /**
     * Constructor
     *
     * @param indicesService service used to get field metadata to validate parameters
     * @param client used to make search requests against the cluster
     */
    public VectorReader(IndicesService indicesService, Client client) {
        this.indicesService = indicesService;
        this.client = client;
    }

    /**
     * Read vectors from a provided index/field and pass them to vectorConsumer that will do something with them.
     *
     * @param indexName name of index containing vectors
     * @param fieldName name of field containing vectors
     * @param maxVectorCount maximum number of vectors to return
     * @param searchSize maximum number of vectors to return in a given search
     * @param vectorConsumer consumer used to do something with the collected vectors after each search
     * @param listener ActionListener that should be called once all search operations complete
     */
    public void read(String indexName, String fieldName, int maxVectorCount, int searchSize,
                     Consumer<List<Float[]>> vectorConsumer, ActionListener<SearchResponse> listener) {

        // Validate arguments
        if (maxVectorCount <= 0) {
            throw new ValidationException();
        }

        if (searchSize > 10000 || searchSize <= 0) {
            throw new ValidationException();
        }

        IndexMetadata indexMetadata = indicesService.clusterService().state().metadata().index(indexName);
        if (indexMetadata == null) {
            throw new ValidationException();
        }

        MappedFieldType fieldType = indicesService.indexServiceSafe(indexMetadata.getIndex()).mapperService()
                .fieldType(fieldName);

        if (!(fieldType instanceof KNNVectorFieldMapper.KNNVectorFieldType)) {
            throw new ValidationException();
        }

        // Start reading vectors from index
        ActionListener<SearchResponse> vectorReaderListener = getVectorReaderListener(fieldName,
                maxVectorCount, 0, vectorConsumer, listener);

        createSearchRequestBuilder(indexName, fieldName, Integer.min(maxVectorCount, searchSize))
                .execute(vectorReaderListener);
    }

    @SuppressWarnings("unchecked")
    private ActionListener<SearchResponse> getVectorReaderListener(String fieldName,
                                                                   final int maxVectorCount,
                                                                   final int collectedVectorCount,
                                                                   Consumer<List<Float[]>> vectorConsumer,
                                                                   ActionListener<SearchResponse> listener) {
        return ActionListener.wrap(searchResponse -> {
            // Get the vectors from the search response
            // Either add the entire set of returned hits, or maxVectorCount - collectedVectorCount hits
            SearchHit[] hits = searchResponse.getHits().getHits();
            int vectorsToAdd = Integer.min(maxVectorCount - collectedVectorCount, hits.length);
            List<Float[]> trainingData = new ArrayList<>();

            for (int i = 0; i < vectorsToAdd; i++) {
                trainingData.add(((List<Double>) hits[i].getSourceAsMap().get(fieldName)).stream()
                        .map(Double::floatValue)
                        .toArray(Float[]::new));
            }

            final int updatedVectorCount = collectedVectorCount + trainingData.size();

            // Do something with the vectors
            vectorConsumer.accept(trainingData);

            if (vectorsToAdd == 0 || updatedVectorCount >= maxVectorCount) {
                listener.onResponse(searchResponse);
            } else {
                ActionListener<SearchResponse> vectorReaderListener = getVectorReaderListener(fieldName, maxVectorCount,
                        updatedVectorCount, vectorConsumer, listener);

                // Create a new search that starts where the last search left off
                createSearchScrollRequestBuilder(searchResponse).execute(vectorReaderListener);
            }
        }, listener::onFailure);
    }

    private SearchRequestBuilder createSearchRequestBuilder(String indexName, String fieldName, int resultSize) {
        ExistsQueryBuilder queryBuilder = new ExistsQueryBuilder(fieldName);

        SearchRequestBuilder searchRequestBuilder = client.prepareSearch(indexName);
        searchRequestBuilder.setScroll(scrollTime);
        searchRequestBuilder.setQuery(queryBuilder);
        searchRequestBuilder.setSize(resultSize);

        // We are only interested in reading vectors from a particular field
        searchRequestBuilder.setFetchSource(fieldName, null);

        return searchRequestBuilder;
    }

    private SearchScrollRequestBuilder createSearchScrollRequestBuilder(SearchResponse searchResponse) {
        SearchScrollRequestBuilder searchScrollRequestBuilder = client.prepareSearchScroll(searchResponse.getScrollId());
        searchScrollRequestBuilder.setScroll(scrollTime);
        return searchScrollRequestBuilder;
    }
}
