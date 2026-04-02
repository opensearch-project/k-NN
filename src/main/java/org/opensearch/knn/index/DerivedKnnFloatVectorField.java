/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;

/**
 * Wrapper around {@link KnnByteVectorField} so that {@link org.opensearch.knn.index.codec.derivedsource.DerivedSourceIndexOperationListener}
 * can understand if it needs to modify the source in the translog
 */
public class DerivedKnnFloatVectorField extends KnnFloatVectorField {

    @Getter
    private final boolean isDerivedEnabled;

    @Getter
    private final float vectorNorm;

    /**
     *
     * @param name Name of the field
     * @param vector vector for the field
     * @param isDerivedEnabled boolean to indicate if derived source is enabled
     */
    public DerivedKnnFloatVectorField(String name, float[] vector, boolean isDerivedEnabled) {
        this(name, vector, isDerivedEnabled, 1.0f);
    }

    public DerivedKnnFloatVectorField(String name, float[] vector, boolean isDerivedEnabled, float norm) {
        super(name, vector);
        this.isDerivedEnabled = isDerivedEnabled;
        this.vectorNorm = norm;
    }

    /**
     *
     * @param name Name of the field
     * @param vector vector for the field
     * @param fieldType FieldType of the field
     * @param isDerivedEnabled boolean to indicate if derived source is enabled
     */
    public DerivedKnnFloatVectorField(String name, float[] vector, FieldType fieldType, boolean isDerivedEnabled) {
        this(name, vector, fieldType, isDerivedEnabled, 1.0f);
    }

    public DerivedKnnFloatVectorField(String name, float[] vector, FieldType fieldType, boolean isDerivedEnabled, float norm) {
        super(name, vector, fieldType);
        this.isDerivedEnabled = isDerivedEnabled;
        this.vectorNorm = norm;
    }
}
