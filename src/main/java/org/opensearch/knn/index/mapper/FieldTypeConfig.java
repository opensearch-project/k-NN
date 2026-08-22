/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Value;
import org.apache.lucene.document.FieldType;

/**
 * Immutable value object holding the field type configuration produced by an {@link EngineFieldStrategy}.
 */
@Value
public class FieldTypeConfig {
    FieldType fieldType;
    FieldType vectorFieldType;
    VectorTransformer vectorTransformer;
    boolean useLuceneBasedVectorField;
}
