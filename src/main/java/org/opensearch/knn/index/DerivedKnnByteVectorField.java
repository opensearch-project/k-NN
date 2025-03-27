/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.Getter;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;

/**
 * Wrapper around {@link KnnByteVectorField} so that {@link org.opensearch.knn.index.codec.derivedsource.DerivedSourceIndexOperationListener}
 * can understand if it needs to modify the source in the translog
 */

public class DerivedKnnByteVectorField extends KnnByteVectorField {

    @Getter
    private final boolean isDerivedEnabled;

    /**
     *
     * @param name Name of the field
     * @param vector vector for the field
     * @param isDerivedEnabled boolean to indicate if derived source is enabled
     */
    public DerivedKnnByteVectorField(String name, byte[] vector, FieldType fieldType, boolean isDerivedEnabled) {
        super(name, vector, fieldType);
        this.isDerivedEnabled = isDerivedEnabled;
    }
}
