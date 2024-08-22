/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.Directory;

@Getter
@AllArgsConstructor
public class QuantizationStateReadConfig {
    private Directory directory;
    private String segmentName;
    private String segmentSuffix;
    private FieldInfo fieldInfo;
}
