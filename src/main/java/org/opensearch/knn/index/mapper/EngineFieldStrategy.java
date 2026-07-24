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

/**
 * Strategy interface for engine-specific field construction and vector field creation.
 * Each KNN engine provides its own implementation to handle field type building,
 * vector field creation, and attribute formatting.
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
     * Creates the list of fields for indexing a float vector.
     * <p>
     * Returns null to signal the caller should use the parent class default field creation path.
     * Override only when the engine requires custom field construction (e.g. Lucene doc-values fields).
     *
     * @param name the field name
     * @param array the float vector
     * @param fieldType the primary field type
     * @param vectorFieldType the doc values field type (may be null)
     * @param stored whether the field is stored
     * @param hasDocValues whether the field has doc values
     * @param isDerivedSourceEnabled whether derived source is enabled
     * @return list of fields to add to the document, or null to use the default field creation path
     */
    default List<Field> createFloatFields(
        String name,
        float[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return null;
    }

    /**
     * Creates the list of fields for indexing a byte vector.
     * <p>
     * Returns null to signal the caller should use the parent class default field creation path.
     * Override only when the engine requires custom field construction (e.g. Lucene doc-values fields).
     *
     * @param name the field name
     * @param array the byte vector
     * @param fieldType the primary field type
     * @param vectorFieldType the doc values field type (may be null)
     * @param stored whether the field is stored
     * @param hasDocValues whether the field has doc values
     * @param isDerivedSourceEnabled whether derived source is enabled
     * @return list of fields to add to the document, or null to use the default field creation path
     */
    default List<Field> createByteFields(
        String name,
        byte[] array,
        FieldType fieldType,
        FieldType vectorFieldType,
        boolean stored,
        boolean hasDocValues,
        boolean isDerivedSourceEnabled
    ) {
        return null;
    }
}
