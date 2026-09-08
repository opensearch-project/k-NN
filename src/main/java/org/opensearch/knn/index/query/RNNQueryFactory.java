/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_DECAY;
import static org.opensearch.knn.common.KNNConstants.MAX_RESULTS_RADIAL_RESCORING;
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
import org.opensearch.knn.index.query.rescore.RescoreContext;

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
    public static Query create(final RNNQueryFactory.CreateQueryRequest createQueryRequest) {
        final String fieldName = createQueryRequest.getFieldName();
        final Float radius = createQueryRequest.getRadius();
        final float[] vector = createQueryRequest.getVector();

        final Query innerQuery;
        if (createQueryRequest.getKnnEngine().createsCustomSegmentFiles()) {
            innerQuery = createNativeEngineRadialQuery(createQueryRequest);
        } else {
            innerQuery = createLuceneRadialQuery(createQueryRequest);
        }

        // SQ or BQ at 1, 2 or 4 bits requires rescoring after radial search to eliminate false positives.
        if (createQueryRequest.getVectorFieldType() != null
            && createQueryRequest.getVectorFieldType().getResolvedSpec().requiresFullPrecisionRadialRescore()) {
            final Query sizeBoundedQuery = createSizeBoundedQuantizedRadialQuery(createQueryRequest);
            if (sizeBoundedQuery != null) {
                return sizeBoundedQuery;
            }
            // Honor the index-level max_result_window setting to cap the candidates rescored.
            // Falls back to MAX_RESULTS_RADIAL_RESCORING if context is unavailable.
            final int fallbackFirstPassK;
            if (createQueryRequest.getContext().isPresent()) {
                fallbackFirstPassK = createQueryRequest.getContext().get().getIndexSettings().getMaxResultWindow();
            } else {
                fallbackFirstPassK = MAX_RESULTS_RADIAL_RESCORING;
            }
            return new RescoreRadialSearchQuery(
                innerQuery,
                fieldName,
                vector,
                radius,
                createQueryRequest.isMemoryOptimizedSearchEnabled(),
                fallbackFirstPassK
            );
        }
        return innerQuery;
    }

    /**
     * Builds the size-bounded form of a quantized radial query: an oversampled top-k first pass of
     * {@code ceil(size * oversample_factor)} candidates instead of an unbounded radial scan, then rescored
     * against full-precision vectors. The cap scales with the oversample factor so a large size can still
     * surface {@link RescoreContext#MAX_FIRST_PASS_RESULTS} hits.
     *
     * @param request the query creation request
     * @return the size-bounded query, or {@code null} if the request size is unavailable
     */
    private static Query createSizeBoundedQuantizedRadialQuery(final CreateQueryRequest request) {
        final Integer size = request.getSize();
        if (size == null || size <= 0) {
            return null;
        }

        final RescoreContext rescoreContext = request.getRescoreContext().orElse(RescoreContext.getDefault());
        final float oversampleFactor = rescoreContext.getOversampleFactor();
        final int firstPassK = (int) Math.min(
            Math.ceil((double) RescoreContext.MAX_FIRST_PASS_RESULTS * oversampleFactor),
            Math.ceil((double) size * oversampleFactor)
        );

        final Query approximateCandidates = KNNQueryFactory.create(
            KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(request.getKnnEngine())
                .indexName(request.getIndexName())
                .fieldName(request.getFieldName())
                .vector(request.getVector())
                .originalVector(request.getOriginalVector())
                .byteVector(request.getByteVector())
                .vectorDataType(request.getVectorDataType())
                .k(firstPassK)
                .methodParameters(request.getMethodParameters())
                .filter(request.getFilter().orElse(null))
                .context(request.getContext().orElse(null))
                .expandNested(request.isExpandNested())
                .memoryOptimizedSearchEnabled(request.isMemoryOptimizedSearchEnabled())
                .build()
        );

        return new RescoreRadialSearchQuery(
            approximateCandidates,
            request.getFieldName(),
            request.getVector(),
            request.getRadius(),
            request.isMemoryOptimizedSearchEnabled(),
            firstPassK
        );
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
     * The graph-traversal buffer decays with {@code DEFAULT_LUCENE_RADIAL_SEARCH_DECAY}
     * so the graph is explored slightly beyond the threshold for better recall, matching the
     * decay-based behavior used on the memory-optimized search (MOS) path.</p>
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
            String.format(
                Locale.ROOT,
                "Creating Lucene r-NN query for index: %s \"\", field: %s \"\", k: %f",
                request.getIndexName(),
                fieldName,
                radius
            )
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
     * If radius is greater than 0, we return {@link FloatVectorSimilarityQuery} which will return all documents with similarity
     * greater than or equal to the resultSimilarity. If filterQuery is not null, it will be used to filter the documents.
     */
    private static Query getFloatVectorSimilarityQuery(
        final String fieldName,
        final float[] floatVector,
        final float resultSimilarity,
        final Query filterQuery
    ) {
        return new FloatVectorSimilarityQuery(fieldName, floatVector, resultSimilarity, DEFAULT_LUCENE_RADIAL_SEARCH_DECAY, filterQuery);
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
        return new ByteVectorSimilarityQuery(fieldName, byteVector, resultSimilarity, DEFAULT_LUCENE_RADIAL_SEARCH_DECAY, filterQuery);
    }
}
