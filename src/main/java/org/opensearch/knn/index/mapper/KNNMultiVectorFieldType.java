/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.lookup.SearchLookup;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.deserializeStoredVector;

@Log4j2
public class KNNMultiVectorFieldType extends MappedFieldType {

    VectorDataType vectorDataType;
    KNNMappingConfig knnMappingConfig;

    /**
     * Constructor for KNNMultiVectorFieldType.
     *
     * @param name name of the field
     * @param metadata metadata of the field
     * @param vectorDataType data type of the vector
     * @param annConfig configuration context for the ANN index
     */
    public KNNMultiVectorFieldType(String name, Map<String, String> metadata, VectorDataType vectorDataType, KNNMappingConfig annConfig) {
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
                    log.warn("Expected type ArrayList for value, but got {} ", value.getClass());
                    return Collections.emptyList();
                }
            }
        };
    }

    @Override
    public String typeName() {
        return KNNMultiVectorFieldMapper.CONTENT_TYPE;
    }

    @Override
    public Query existsQuery(QueryShardContext context) {
        return new FieldExistsQuery(name());
    }

    @Override
    public Query termQuery(Object o, QueryShardContext context) {
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
        //TODO
        return deserializeStoredVector((BytesRef) value, vectorDataType);
    }

    public int getVectorDimensions() {
        return knnMappingConfig.getDimension();
    }

    public VectorDataType getVectorDataType() {
        return vectorDataType;
    }
}
