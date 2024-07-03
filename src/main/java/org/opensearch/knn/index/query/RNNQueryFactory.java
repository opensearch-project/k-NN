/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

import java.util.Locale;
import java.util.Map;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

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
        final byte[] byteVector = createQueryRequest.getByteVector();
        final VectorDataType vectorDataType = createQueryRequest.getVectorDataType();
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

        log.debug(String.format("Creating Lucene r-NN query for index: %s \"\", field: %s \"\", k: %f", indexName, fieldName, radius));
        switch (vectorDataType) {
            case BYTE:
                return getByteVectorSimilarityQuery(fieldName, byteVector, radius, filterQuery);
            case FLOAT:
                return getFloatVectorSimilarityQuery(fieldName, vector, radius, filterQuery);
            default:
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s], but got: %s",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES,
                        vectorDataType
                    )
                );
        }
    }

    /**
     * If radius is greater than 0, we return {@link FloatVectorSimilarityQuery} which will return all documents with similarity
     * greater than or equal to the resultSimilarity. If filterQuery is not null, it will be used to filter the documents.
     */
    private static Query getFloatVectorSimilarityQuery(
        final String fieldName,
        final float[] floatVector,
        final float resultSimilarity,
        final Query filterQuery
    ) {
        return new FloatVectorSimilarityQuery(
            fieldName,
            floatVector,
            DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * resultSimilarity,
            resultSimilarity,
            filterQuery
        );
    }

    /**
     * If radius is greater than 0, we return {@link ByteVectorSimilarityQuery} which will return all documents with similarity
     * greater than or equal to the resultSimilarity. If filterQuery is not null, it will be used to filter the documents.
     */
    private static Query getByteVectorSimilarityQuery(
        final String fieldName,
        final byte[] byteVector,
        final float resultSimilarity,
        final Query filterQuery
    ) {
        return new ByteVectorSimilarityQuery(
            fieldName,
            byteVector,
            DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * resultSimilarity,
            resultSimilarity,
            filterQuery
        );
    }
}
