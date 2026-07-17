/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;

/**
 * Immutable value object holding the field type configuration produced by an {@link EngineFieldStrategy}.
 */
public final class FieldTypeConfig {
    private final FieldType fieldType;
    private final FieldType vectorFieldType;
    private final VectorTransformer vectorTransformer;
    private final boolean useLuceneBasedVectorField;

    public FieldTypeConfig(
        FieldType fieldType,
        FieldType vectorFieldType,
        VectorTransformer vectorTransformer,
        boolean useLuceneBasedVectorField
    ) {
        this.fieldType = fieldType;
        this.vectorFieldType = vectorFieldType;
        this.vectorTransformer = vectorTransformer;
        this.useLuceneBasedVectorField = useLuceneBasedVectorField;
    }

    public FieldType getFieldType() {
        return fieldType;
    }

    public FieldType getVectorFieldType() {
        return vectorFieldType;
    }

    public VectorTransformer getVectorTransformer() {
        return vectorTransformer;
    }

    public boolean isUseLuceneBasedVectorField() {
        return useLuceneBasedVectorField;
    }
}
