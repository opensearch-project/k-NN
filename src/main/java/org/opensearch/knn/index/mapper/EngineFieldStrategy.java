/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.List;
import java.util.Optional;

/**
 * Strategy interface for engine-specific field type construction. Each KNN engine provides its
 * own implementation to handle field type building and attribute formatting. Engines that require
 * custom vector field creation override {@link #getFloatFieldOverrides} and
 * {@link #getByteFieldOverrides}; otherwise the default field creation path owned by
 * {@link KNNVectorFieldMapper} is used.
 */
public interface EngineFieldStrategy {

    /**
     * Builds the field type configuration for this engine.
     *
     * @param knnMappingConfig the mapping configuration
     * @param resolvedKnnMethodContext the resolved method context
     * @param knnLibraryIndexingContext the library indexing context
     * @param vectorDataType the vector data type
     * @param indexCreatedVersion the version when the index was created
     * @param hasDocValues whether the field has doc values
     * @return a FieldTypeConfig containing the field type, vector field type, and transformer
     */
    FieldTypeConfig buildFieldTypeConfig(
        KNNMappingConfig knnMappingConfig,
        KNNMethodContext resolvedKnnMethodContext,
        KNNLibraryIndexingContext knnLibraryIndexingContext,
        VectorDataType vectorDataType,
        Version indexCreatedVersion,
        boolean hasDocValues
    );

    /**
     * Returns engine-specific fields for indexing a float vector, or {@link Optional#empty()} to
     * use the default field creation path owned by {@link KNNVectorFieldMapper}.
     *
     * @param name the field name
     * @param array the float vector
     * @param fieldType the primary field type
     * @param vectorFieldType the doc values field type (may be null)
     * @param stored whether the field is stored
     * @param hasDocValues whether the field has doc values
     * @param isDerivedSourceEnabled whether derived source is enabled
     * @return optional list of fields to add to the document
     */
    default Optional<List<Field>> getFloatFieldOverrides(
        String name,
        float[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return Optional.empty();
    }

    /**
     * Returns engine-specific fields for indexing a byte vector, or {@link Optional#empty()} to
     * use the default field creation path owned by {@link KNNVectorFieldMapper}.
     *
     * @param name the field name
     * @param array the byte vector
     * @param fieldType the primary field type
     * @param vectorFieldType the doc values field type (may be null)
     * @param stored whether the field is stored
     * @param hasDocValues whether the field has doc values
     * @param isDerivedSourceEnabled whether derived source is enabled
     * @return optional list of fields to add to the document
     */
    default Optional<List<Field>> getByteFieldOverrides(
        String name,
        byte[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return Optional.empty();
    }
}
