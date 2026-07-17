/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import static org.opensearch.knn.common.KNNConstants.MAX_RESULTS_RADIAL_RESCORING;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

import java.util.Locale;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

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

        log.info("[RADIAL-DEBUG] RNNQueryFactory.create: field={}, radius={}, engine={}, MOS={}",
            fieldName, radius, createQueryRequest.getKnnEngine(), createQueryRequest.isMemoryOptimizedSearchEnabled());

        final Query innerQuery;
        if (createQueryRequest.isMemoryOptimizedSearchEnabled()) {
            // MOS (Faiss HNSW via Lucene's vector reader): use the unified seeded radial search.
            log.info("[RADIAL-DEBUG] Routing to createSeededRadialQuery (MOS path)");
            innerQuery = createSeededRadialQuery(createQueryRequest);
        } else if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            log.info("[RADIAL-DEBUG] Routing to createNativeEngineRadialQuery (native FAISS/NMSLIB path)");
            innerQuery = createNativeEngineRadialQuery(createQueryRequest);
        } else {
            log.info("[RADIAL-DEBUG] Routing to createLuceneRadialQuery (Lucene engine path)");
            innerQuery = createLuceneRadialQuery(createQueryRequest);
        }

        if (createQueryRequest.getVectorFieldType() != null && createQueryRequest.getVectorFieldType().isRescoringRequiredForRadial()) {
            // Honor the index-level max_result_window setting to cap the number of results retained
            // after rescoring. Falls back to MAX_RESULTS_RADIAL_RESCORING if context is unavailable.
            final int maxResultsSize;
            if (createQueryRequest.getContext().isPresent()) {
                maxResultsSize = createQueryRequest.getContext().get().getIndexSettings().getMaxResultWindow();
            } else {
                maxResultsSize = MAX_RESULTS_RADIAL_RESCORING;
            }
            log.info("[RADIAL-DEBUG] Wrapping with RescoreRadialSearchQuery: radius={}, MOS={}, maxResultsSize={}",
                radius, createQueryRequest.isMemoryOptimizedSearchEnabled(), maxResultsSize);
            return new RescoreRadialSearchQuery(
                innerQuery,
                fieldName,
                vector,
                radius,
                createQueryRequest.isMemoryOptimizedSearchEnabled(),
                maxResultsSize
            );
        }
        return innerQuery;
    }

    /**
     * Creates a {@link RadialSearchQuery} for memory-optimized search (Faiss HNSW via Lucene's vector reader).
     * <p>
     * This uses the same two-phase seeded radial search as the Lucene engine path, since both
     * ultimately search via {@code LeafReader.searchNearestVectors()}.
     *
     * @param request the query creation request containing all parameters
     * @return a {@link RadialSearchQuery} configured for seeded radius-based search
     */
    private static Query createSeededRadialQuery(CreateQueryRequest request) {
        final String fieldName = request.getFieldName();
        final Float radius = request.getRadius();
        final Query filterQuery = getFilterQuery(request);
        final int efSearch = resolveEfSearch(request);

        log.info("[RADIAL-DEBUG] createSeededRadialQuery: index={}, field={}, radius={}, efSearch={}",
            request.getIndexName(), fieldName, radius, efSearch);

        return new RadialSearchQuery(fieldName, request.getVector(), radius, efSearch, filterQuery);
    }

    /**
     * Creates a {@link KNNQuery} for native engines (Faiss, NMSLIB) that use custom segment files.
     *
     * <p>The returned query carries the radius threshold and is executed via JNI through
     * {@code JNIService.radiusQueryIndex()} in {@code DefaultKNNWeight}.</p>
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
     * Creates a radial search query for engines that do not use custom segment files (i.e., Lucene engine).
     * <p>
     * For float vectors, uses {@link RadialSearchQuery} which performs a two-phase seeded radial
     * search via Lucene's HNSW graph. For byte vectors, falls back to {@link ByteVectorSimilarityQuery}.
     *
     * @param request the query creation request containing all parameters
     * @return a query configured for radius-based search
     * @throws IllegalArgumentException if the vector data type is not supported
     */
    private static Query createLuceneRadialQuery(CreateQueryRequest request) {
        final String fieldName = request.getFieldName();
        final Float radius = request.getRadius();
        final Query filterQuery = getFilterQuery(request);

        log.info("[RADIAL-DEBUG] createLuceneRadialQuery: index={}, field={}, radius={}, efSearch={}",
            request.getIndexName(), fieldName, radius, resolveEfSearch(request));

        switch (request.getVectorDataType()) {
            case BYTE:
                return new RadialSearchQuery(fieldName, request.getByteVector(), radius, resolveEfSearch(request), filterQuery);
            case FLOAT:
                return new RadialSearchQuery(fieldName, request.getVector(), radius, resolveEfSearch(request), filterQuery);
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

    private static int resolveEfSearch(final CreateQueryRequest request) {
        try {
            return IndexHyperParametersUtil.getHNSWEFSearchValue(request.getMethodParameters(), request.getIndexName());
        } catch (Exception e) {
            return KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH;
        }
    }
}
