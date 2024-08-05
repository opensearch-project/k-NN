/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Map;

/**
 * Class to create radius nearest neighbor queries
 */
@Log4j2
public class RNNQueryFactory extends BaseQueryFactory {

    /**
     * Creates a Lucene query for a particular engine.
     *
     * @param knnEngine Engine to create the query for
     * @param indexName Name of the OpenSearch index that is being queried
     * @param fieldName Name of the field in the OpenSearch index that will be queried
     * @param vector The query vector to get the nearest neighbors for
     * @param radius the radius threshold for the nearest neighbors
     * @return Lucene Query
     */
    public static Query create(
        KNNEngine knnEngine,
        String indexName,
        String fieldName,
        float[] vector,
        Float radius,
        VectorDataType vectorDataType
    ) {
        final CreateQueryRequest createQueryRequest = CreateQueryRequest.builder()
            .knnEngine(knnEngine)
            .indexName(indexName)
            .fieldName(fieldName)
            .vector(vector)
            .vectorDataType(vectorDataType)
            .radius(radius)
            .build();
        return create(createQueryRequest);
    }

    /**
     * Creates a Lucene query for a particular engine.
     * @param createQueryRequest request object that has all required fields to construct the query
     * @return Lucene Query
     */
    public static Query create(RNNQueryFactory.CreateQueryRequest createQueryRequest) {
        final String indexName = createQueryRequest.getIndexName();
        final String fieldName = createQueryRequest.getFieldName();
        final Float radius = createQueryRequest.getRadius();
        final float[] vector = createQueryRequest.getVector();
        final Query filterQuery = getFilterQuery(createQueryRequest);
        final Map<String, ?> methodParameters = createQueryRequest.getMethodParameters();

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            BitSetProducer parentFilter = null;
            QueryShardContext context = createQueryRequest.getContext().get();

            if (createQueryRequest.getContext().isPresent()) {
                parentFilter = context.getParentFilter();
            }
            IndexSettings indexSettings = context.getIndexSettings();
            KNNQuery.Context knnQueryContext = new KNNQuery.Context(indexSettings.getMaxResultWindow());

            return KNNQuery.builder()
                .field(fieldName)
                .queryVector(vector)
                .indexName(indexName)
                .parentsFilter(parentFilter)
                .radius(radius)
                .methodParameters(methodParameters)
                .context(knnQueryContext)
                .filterQuery(filterQuery)
                .build();
        }
        throw new IllegalStateException("Radial search is supported only with faiss Engine");
    }
}
