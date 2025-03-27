/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.opensearch.knn.index.mapper.DerivedSourceFieldType;

/**
 * Wrapper around {@link KnnByteVectorField} so that {@link org.opensearch.knn.index.codec.derivedsource.DerivedSourceIndexOperationListener}
 * can understand if it needs to modify the source in the translog
 */
public class DerivedKnnByteVectorField extends KnnByteVectorField implements DerivedSourceFieldType {

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

    @Override
    public boolean isDerivedSourceEnabled() {
        return isDerivedEnabled;
    }
}
