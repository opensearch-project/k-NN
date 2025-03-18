/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Builder;
import org.apache.lucene.index.FieldInfo;

/**
 * Light wrapper around {@link FieldInfo} that indicates whether the field is nested or not.
 */
@Builder
public record DerivedFieldInfo(FieldInfo fieldInfo, boolean isNested) {
    /**
     * @return name of the field
     */
    public String name() {
        return fieldInfo.name;
    }
}
