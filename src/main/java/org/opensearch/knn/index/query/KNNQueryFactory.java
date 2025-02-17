/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.index.query.QueryShardContext;
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
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;

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
        final byte[] byteVector = createQueryRequest.getByteVector();
        final VectorDataType vectorDataType = createQueryRequest.getVectorDataType();
        final Query filterQuery = getFilterQuery(createQueryRequest);
        final Map<String, ?> methodParameters = createQueryRequest.getMethodParameters();
        final RescoreContext rescoreContext = createQueryRequest.getRescoreContext().orElse(null);
        final KNNEngine knnEngine = createQueryRequest.getKnnEngine();
        final boolean expandNested = createQueryRequest.getExpandNested().orElse(false);
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

            KNNQuery knnQuery = null;
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
                        .build();
                    break;
                default:
                    knnQuery = KNNQuery.builder()
                        .field(fieldName)
                        .queryVector(vector)
                        .indexName(indexName)
                        .parentsFilter(parentFilter)
                        .k(k)
                        .methodParameters(methodParameters)
                        .filterQuery(validatedFilterQuery)
                        .vectorDataType(vectorDataType)
                        .rescoreContext(rescoreContext)
                        .shardId(shardId)
                        .build();
            }

            return new NativeEngineKnnVectorQuery(knnQuery, QueryUtils.INSTANCE, expandNested);
        }

        Integer requestEfSearch = null;
        if (methodParameters != null && methodParameters.containsKey(METHOD_PARAMETER_EF_SEARCH)) {
            requestEfSearch = (Integer) methodParameters.get(METHOD_PARAMETER_EF_SEARCH);
        }
        int luceneK = requestEfSearch == null ? k : Math.max(k, requestEfSearch);
        log.debug("Creating Lucene k-NN query for index: {}, field:{}, k: {}", indexName, fieldName, k);
        switch (vectorDataType) {
            case BYTE:
            case BINARY:
                return new LuceneEngineKnnVectorQuery(
                    getKnnByteVectorQuery(fieldName, byteVector, luceneK, filterQuery, parentFilter, expandNested)
                );
            case FLOAT:
                return new LuceneEngineKnnVectorQuery(
                    getKnnFloatVectorQuery(fieldName, vector, luceneK, filterQuery, parentFilter, expandNested)
                );
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

    private static Query validateFilterQuerySupport(final Query filterQuery, final KNNEngine knnEngine) {
        log.debug("filter query {}, knnEngine {}", filterQuery, knnEngine);
        if (filterQuery != null && KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            return filterQuery;
        }
        return null;
    }

    /**
     * If parentFilter is not null, it is a nested query. Therefore, we delegate creation of query to {@link NestedKnnVectorQueryFactory}
     * which will create query to dedupe search result per parent so that we can get k parent results at the end.
     */
    private static Query getKnnByteVectorQuery(
        final String fieldName,
        final byte[] byteVector,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNested
    ) {
        if (parentFilter == null) {
            assert expandNested == false : "expandNested is allowed to be true only for nested fields.";
            return new KnnByteVectorQuery(fieldName, byteVector, k, filterQuery);
        } else {
            return NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                byteVector,
                k,
                filterQuery,
                parentFilter,
                expandNested
            );
        }
    }

    /**
     * If parentFilter is not null, it is a nested query. Therefore, we delegate creation of query to {@link NestedKnnVectorQueryFactory}
     * which will create query to dedupe search result per parent so that we can get k parent results at the end.
     */
    private static Query getKnnFloatVectorQuery(
        final String fieldName,
        final float[] floatVector,
        final int k,
        final Query filterQuery,
        final BitSetProducer parentFilter,
        final boolean expandNested
    ) {
        if (parentFilter == null) {
            assert expandNested == false : "expandNested is allowed to be true only for nested fields.";
            return new KnnFloatVectorQuery(fieldName, floatVector, k, filterQuery);
        } else {
            return NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                floatVector,
                k,
                filterQuery,
                parentFilter,
                expandNested
            );
        }
    }
}
