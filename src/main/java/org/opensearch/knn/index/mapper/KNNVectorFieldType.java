/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import org.apache.lucene.search.DocValuesFieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.Nullable;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.lookup.SearchLookup;

import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.deserializeStoredVector;

/**
 * A KNNVector field type to represent the vector field in Opensearch
 */
@Getter
public class KNNVectorFieldType extends MappedFieldType {
    int dimension;
    String modelId;
    KNNMethodContext knnMethodContext;
    VectorDataType vectorDataType;
    SpaceType spaceType;

    public KNNVectorFieldType(String name, Map<String, String> meta, int dimension, VectorDataType vectorDataType, SpaceType spaceType) {
        this(name, meta, dimension, null, null, vectorDataType, spaceType);
    }

    public KNNVectorFieldType(String name, Map<String, String> meta, int dimension, KNNMethodContext knnMethodContext) {
        this(name, meta, dimension, knnMethodContext, null, DEFAULT_VECTOR_DATA_TYPE_FIELD, knnMethodContext.getSpaceType());
    }

    public KNNVectorFieldType(String name, Map<String, String> meta, int dimension, KNNMethodContext knnMethodContext, String modelId) {
        this(name, meta, dimension, knnMethodContext, modelId, DEFAULT_VECTOR_DATA_TYPE_FIELD, null);
    }

    public KNNVectorFieldType(
        String name,
        Map<String, String> meta,
        int dimension,
        KNNMethodContext knnMethodContext,
        VectorDataType vectorDataType
    ) {
        this(name, meta, dimension, knnMethodContext, null, vectorDataType, knnMethodContext.getSpaceType());
    }

    public KNNVectorFieldType(
        String name,
        Map<String, String> meta,
        int dimension,
        @Nullable KNNMethodContext knnMethodContext,
        @Nullable String modelId,
        VectorDataType vectorDataType,
        @Nullable SpaceType spaceType
    ) {
        super(name, false, false, true, TextSearchInfo.NONE, meta);
        this.dimension = dimension;
        this.modelId = modelId;
        this.knnMethodContext = knnMethodContext;
        this.vectorDataType = vectorDataType;
        this.spaceType = spaceType;
    }

    @Override
    public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
        throw new UnsupportedOperationException("KNN Vector do not support fields search");
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
}
