/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.DocValuesFieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.lookup.SearchLookup;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.deserializeStoredVector;

/**
 * A KNNVector field type to represent the vector field in Opensearch
 */
@Getter
public class KNNVectorFieldType extends MappedFieldType {
    private static final Logger logger = LogManager.getLogger(KNNVectorFieldType.class);
    KNNMappingConfig knnMappingConfig;
    VectorDataType vectorDataType;

    /**
     * Constructor for KNNVectorFieldType.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     */
    public KNNVectorFieldType(String name, Map<String, String> metadata, VectorDataType vectorDataType, KNNMappingConfig annConfig) {
        super(name, false, false, true, TextSearchInfo.NONE, metadata);
        this.vectorDataType = vectorDataType;
        this.knnMappingConfig = annConfig;
    }

    @Override
    public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
        return new ArraySourceValueFetcher(name(), context) {
            @Override
            protected Object parseSourceValue(Object value) {
                if (value instanceof ArrayList) {
                    return value;
                } else {
                    logger.warn("Expected type ArrayList for value, but got {} ", value.getClass());
                    return Collections.emptyList();
                }
            }
        };
    }

    @Override
    public String typeName() {
        return KNNVectorFieldMapper.CONTENT_TYPE;
    }

    @Override
    public Query existsQuery(QueryShardContext context) {
        return new DocValuesFieldExistsQuery(name());
    }

    @Override
    public Query termQuery(Object value, QueryShardContext context) {
        throw new QueryShardException(
            context,
            String.format(Locale.ROOT, "KNN vector do not support exact searching, use KNN queries instead: [%s]", name())
        );
    }

    @Override
    public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
        failIfNoDocValues();
        return new KNNVectorIndexFieldData.Builder(name(), CoreValuesSourceType.BYTES, this.vectorDataType);
    }

    @Override
    public Object valueForDisplay(Object value) {
        return deserializeStoredVector((BytesRef) value, vectorDataType);
    }

    /**
     * Resolve the rescore context provided for a user based on the field configuration
     *
     * @param userProvidedContext {@link RescoreContext} user passed; if null, the default should be configured
     * @return resolved {@link RescoreContext}
     */
    public RescoreContext resolveRescoreContext(RescoreContext userProvidedContext) {
        if (userProvidedContext != null) {
            return userProvidedContext;
        }
        KNNMappingConfig knnMappingConfig = getKnnMappingConfig();
        int dimension = knnMappingConfig.getDimension();
        CompressionLevel compressionLevel = knnMappingConfig.getCompressionLevel();
        Mode mode = knnMappingConfig.getMode();
        return compressionLevel.getDefaultRescoreContext(mode, dimension);
    }
}
