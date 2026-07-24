/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.knn.index.DerivedKnnByteVectorField;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.buildDocValuesFieldType;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;

/**
 * Lucene engine implementation of {@link EngineFieldStrategy}.
 * Handles field type construction and vector field creation for the Lucene KNN engine.
 */
public final class LuceneFieldStrategy implements EngineFieldStrategy {

    public static final LuceneFieldStrategy INSTANCE = new LuceneFieldStrategy();

    private LuceneFieldStrategy() {}

    @Override
    public FieldTypeConfig buildFieldTypeConfig(
        KNNMappingConfig knnMappingConfig,
        KNNMethodContext resolvedKnnMethodContext,
        KNNLibraryIndexingContext knnLibraryIndexingContext,
        VectorDataType vectorDataType,
        Version indexCreatedVersion,
        boolean hasDocValues
    ) {
        KNNVectorSimilarityFunction knnVectorSimilarityFunction = resolvedKnnMethodContext.getSpaceType().getKnnVectorSimilarityFunction();

        FieldType fieldType = vectorDataType.createKnnVectorFieldType(knnMappingConfig.getDimension(), knnVectorSimilarityFunction);

        FieldType vectorFieldType;
        if (hasDocValues) {
            vectorFieldType = buildDocValuesFieldType(resolvedKnnMethodContext.getKnnEngine());
        } else {
            vectorFieldType = null;
        }

        return new FieldTypeConfig(fieldType, vectorFieldType, null, false);
    }

    @Override
    public List<Field> createFloatFields(
        String name,
        float[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        final List<Field> fields = new ArrayList<>();
        fields.add(new DerivedKnnFloatVectorField(name, array, fieldType, isDerivedSourceEnabled));
        if (hasDocValues && vectorFieldType != null) {
            fields.add(new VectorField(name, array, vectorFieldType));
        }
        if (stored) {
            fields.add(createStoredFieldForFloatVector(name, array));
        }
        return fields;
    }

    @Override
    public List<Field> createByteFields(
        String name,
        byte[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        final List<Field> fields = new ArrayList<>();
        fields.add(new DerivedKnnByteVectorField(name, array, fieldType, isDerivedSourceEnabled));
        if (hasDocValues && vectorFieldType != null) {
            fields.add(new VectorField(name, array, vectorFieldType));
        }
        if (stored) {
            fields.add(createStoredFieldForByteVector(name, array));
        }
        return fields;
    }
}
