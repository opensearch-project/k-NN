/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.lucenelib.NestedKnnVectorQueryFactory;
import org.opensearch.knn.index.query.lucene.LuceneEngineKnnVectorQuery;
import org.opensearch.knn.index.query.nativelib.NativeEngineKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_NESTED_FIELDS;

/**
 * Creates the Lucene k-NN queries
 */
@Log4j2
public class KNNQueryFactory extends BaseQueryFactory {
    /**
     * Creates a Lucene query for a particular engine.
     * @param createQueryRequest request object that has all required fields to construct the query
     * @return Lucene Query
     */
    public static Query create(CreateQueryRequest createQueryRequest) {
        // Engines that create their own custom segment files cannot use the Lucene's KnnVectorQuery. They need to
        // use the custom query type created by the plugin
        final String indexName = createQueryRequest.getIndexName();
        final String fieldName = createQueryRequest.getFieldName();
        final int k = createQueryRequest.getK();
        final float[] vector = createQueryRequest.getVector();
        final float[] originalVector = createQueryRequest.getOriginalVector();
        final byte[] byteVector = createQueryRequest.getByteVector();
        final VectorDataType vectorDataType = createQueryRequest.getVectorDataType();
        final Query filterQuery = getFilterQuery(createQueryRequest);
        final Map<String, ?> methodParameters = createQueryRequest.getMethodParameters();
        final RescoreContext rescoreContext = createQueryRequest.getRescoreContext().orElse(null);
        final boolean expandNested = createQueryRequest.isExpandNested();
        final boolean memoryOptimizedSearchEnabled = createQueryRequest.isMemoryOptimizedSearchEnabled();

        BitSetProducer parentFilter = null;
        int shardId = -1;
        if (createQueryRequest.getContext().isPresent()) {
            QueryShardContext context = createQueryRequest.getContext().get();
            parentFilter = context.getParentFilter();
            shardId = context.getShardId();
        }

        if (parentFilter == null && expandNested) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for the [%s] field. [%s] is only supported with a nested field.",
                    EXPAND_NESTED,
                    EXPAND_NESTED
                )
            );
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(createQueryRequest.getKnnEngine())) {
            final Query validatedFilterQuery = validateFilterQuerySupport(filterQuery, createQueryRequest.getKnnEngine());

            log.debug(
                "Creating custom k-NN query for index:{}, field:{}, k:{}, filterQuery:{}, efSearch:{}",
                indexName,
                fieldName,
                k,
                validatedFilterQuery,
                methodParameters
            );

            final KNNQuery knnQuery;
            switch (vectorDataType) {
                case BINARY:
                    knnQuery = KNNQuery.builder()
                        .field(fieldName)
                        .byteQueryVector(byteVector)
                        .indexName(indexName)
                        .parentsFilter(parentFilter)
                        .k(k)
                        .methodParameters(methodParameters)
                        .filterQuery(validatedFilterQuery)
                        .vectorDataType(vectorDataType)
                        .rescoreContext(rescoreContext)
                        .shardId(shardId)
                        .isMemoryOptimizedSearch(memoryOptimizedSearchEnabled)
                        .build();
                    break;
                default:
                    knnQuery = KNNQuery.builder()
                        .field(fieldName)
                        .queryVector(vector)
                        .originalQueryVector(originalVector)
                        .byteQueryVector(byteVector)
                        .indexName(indexName)
                        .parentsFilter(parentFilter)
                        .k(k)
                        .methodParameters(methodParameters)
                        .filterQuery(validatedFilterQuery)
                        .vectorDataType(vectorDataType)
                        .rescoreContext(rescoreContext)
                        .shardId(shardId)
                        .isMemoryOptimizedSearch(memoryOptimizedSearchEnabled)
                        .build();
            }

            if (memoryOptimizedSearchEnabled
                || createQueryRequest.getRescoreContext().isPresent()
                || (ENGINES_SUPPORTING_NESTED_FIELDS.contains(createQueryRequest.getKnnEngine()) && expandNested)) {
                return new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.getInstance(), expandNested);
            }

            return knnQuery;
        }

        int overSampledK = k;
        boolean needsRescore = shouldRescore(rescoreContext);
        if (needsRescore) {
            // Will always do shard level rescoring whenever rescore is required.
            overSampledK = rescoreContext.getFirstPassK(k, false, getDimension(vector, byteVector));
        }

        int luceneK = Math.max(overSampledK, getEfSearch(methodParameters, indexName));
        log.debug("Creating Lucene k-NN query for index: {}, field:{}, k: {}", indexName, fieldName, luceneK);
        Query luceneKnnQuery = new LuceneEngineKnnVectorQuery(
            getKnnVectorQuery(fieldName, vector, byteVector, luceneK, filterQuery, parentFilter, expandNested, vectorDataType),
            luceneK,
            k
        );
        return needsRescore ? new RescoreKNNVectorQuery(luceneKnnQuery, fieldName, k, vector, shardId) : luceneKnnQuery;

    }

    // Determine the ef_search value using the following priority order:
    // 1. Use ef_search from method parameters if specified in the query
    // 2. Otherwise, use ef_search from index setting (knn.algo_param.ef_search)
    // 3. If neither exists, fall back to default ef_search value based on index version
    private static int getEfSearch(final Map<String, ?> methodParameters, final String indexName) {
        if (methodParameters != null && methodParameters.containsKey(METHOD_PARAMETER_EF_SEARCH)) {
            return (Integer) methodParameters.get(METHOD_PARAMETER_EF_SEARCH);
        }

        // Returns ef_search from index setting (knn.algo_param.ef_search) or
        // falls back to default ef_search value based on index version
        return KNNSettings.getEfSearchParam(indexName);
    }

    private static int getDimension(float[] floatQueryVector, byte[] byteQueryVector) {
        if (floatQueryVector != null) {
            return floatQueryVector.length;
        }
        if (byteQueryVector != null) {
            return byteQueryVector.length;
        }
        throw new IllegalStateException("QueryVector has neither float nor byte array");
    }

    private static Query validateFilterQuerySupport(final Query filterQuery, final KNNEngine knnEngine) {
        log.debug("filter query {}, knnEngine {}", filterQuery, knnEngine);
        if (filterQuery != null && KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            return filterQuery;
        }
        return null;
    }

    private static boolean shouldRescore(RescoreContext rescoreContext) {
        return rescoreContext != null && rescoreContext.isRescoreEnabled();
    }

    private static Query getKnnVectorQuery(
        final String fieldName,
        final float[] floatQueryVector,
        final byte[] byteQueryVector,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNested,
        @NonNull final VectorDataType vectorDataType
    ) {
        if (parentFilter == null) {
            assert expandNested == false : "expandNested is allowed to be true only for nested fields.";
            return vectorDataType == VectorDataType.FLOAT
                ? new KnnFloatVectorQuery(fieldName, floatQueryVector, k, filterQuery)
                : new KnnByteVectorQuery(fieldName, byteQueryVector, k, filterQuery);
        }
        // If parentFilter is not null, it is a nested query. Therefore, we delegate creation of query to {@link
        // NestedKnnVectorQueryFactory}
        // which will create query to dedupe search result per parent so that we can get k parent results at the end.
        return vectorDataType == VectorDataType.FLOAT
            ? NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                floatQueryVector,
                k,
                filterQuery,
                parentFilter,
                expandNested
            )
            : NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                byteQueryVector,
                k,
                filterQuery,
                parentFilter,
                expandNested
            );
    }
}
