/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

import java.util.Locale;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.ByteVectorSimilarityQuery;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;

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
        final String fieldName = createQueryRequest.getFieldName();
        final Float radius = createQueryRequest.getRadius();
        final float[] vector = createQueryRequest.getVector();

        final Query innerQuery;
        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            innerQuery = createNativeEngineRadialQuery(createQueryRequest);
        } else {
            innerQuery = createLuceneRadialQuery(createQueryRequest);
        }

        if (isQuantizedForRescore(createQueryRequest)) {
            return new RescoreRadialSearchQuery(innerQuery, fieldName, vector, radius);
        }
        return innerQuery;
    }

    /**
     * Creates a {@link KNNQuery} for native engines (Faiss, NMSLIB) that use custom segment files.
     *
     * <p>The returned query carries the radius threshold and is executed via JNI through
     * {@code JNIService.radiusQueryIndex()} in {@code DefaultKNNWeight}, or via
     * {@code RadiusVectorSimilarityCollector} in {@code MemoryOptimizedKNNWeight}.</p>
     *
     * @param request the query creation request containing all parameters
     * @return a {@link KNNQuery} configured for radius-based search
     */
    private static Query createNativeEngineRadialQuery(CreateQueryRequest request) {
        BitSetProducer parentFilter = null;
        QueryShardContext context = request.getContext().get();

        if (request.getContext().isPresent()) {
            parentFilter = context.getParentFilter();
        }
        IndexSettings indexSettings = context.getIndexSettings();
        KNNQuery.Context knnQueryContext = new KNNQuery.Context(indexSettings.getMaxResultWindow());

        return KNNQuery.builder()
            .field(request.getFieldName())
            .queryVector(request.getVector())
            .originalQueryVector(request.getOriginalVector())
            .byteQueryVector(request.getByteVector())
            .indexName(request.getIndexName())
            .parentsFilter(parentFilter)
            .radius(request.getRadius())
            .vectorDataType(request.getVectorDataType())
            .methodParameters(request.getMethodParameters())
            .context(knnQueryContext)
            .isMemoryOptimizedSearch(request.isMemoryOptimizedSearchEnabled())
            .filterQuery(getFilterQuery(request))
            .build();
    }

    /**
     * Creates a Lucene-native radial search query ({@link FloatVectorSimilarityQuery} or
     * {@link ByteVectorSimilarityQuery}) for engines that do not use custom segment files.
     *
     * <p>These queries use Lucene's built-in HNSW graph traversal with a similarity threshold.
     * The traversal similarity is set to {@code 0.95 * resultSimilarity} to allow the graph
     * to be explored slightly beyond the threshold for better recall.</p>
     *
     * @param request the query creation request containing all parameters
     * @return a Lucene similarity query configured for radius-based search
     * @throws IllegalArgumentException if the vector data type is not supported
     */
    private static Query createLuceneRadialQuery(CreateQueryRequest request) {
        final String fieldName = request.getFieldName();
        final Float radius = request.getRadius();
        final Query filterQuery = getFilterQuery(request);

        log.debug(
            String.format("Creating Lucene r-NN query for index: %s \"\", field: %s \"\", k: %f", request.getIndexName(), fieldName, radius)
        );

        switch (request.getVectorDataType()) {
            case BYTE:
                return getByteVectorSimilarityQuery(fieldName, request.getByteVector(), radius, filterQuery);
            case FLOAT:
                return getFloatVectorSimilarityQuery(fieldName, request.getVector(), radius, filterQuery);
            default:
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s], but got: %s",
                        VECTOR_DATA_TYPE_FIELD,
                        SUPPORTED_VECTOR_DATA_TYPES,
                        request.getVectorDataType()
                    )
                );
        }
    }

    /**
     * Determines whether the index uses 32x scalar quantization, which requires rescoring
     * after radial search to eliminate false positives from quantization error.
     */
    private static boolean isQuantizedForRescore(CreateQueryRequest request) {
        return request.getCompressionLevel() == CompressionLevel.x32;
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
