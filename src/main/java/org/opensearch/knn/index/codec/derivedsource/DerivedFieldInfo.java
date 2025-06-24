/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Builder;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.mapper.VectorTransformer;

/**
 * Light wrapper around {@link FieldInfo} that indicates whether the field is nested or not.
 */
@Builder
public record DerivedFieldInfo(FieldInfo fieldInfo, boolean isNested, VectorTransformer vectorTransformer) {
    /**
     * @return name of the field
     */
    public String name() {
        return fieldInfo.name;
    }
}
