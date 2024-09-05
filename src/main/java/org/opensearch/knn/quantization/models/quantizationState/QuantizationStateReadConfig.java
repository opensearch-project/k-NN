/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

@Getter
@AllArgsConstructor
public class QuantizationStateReadConfig {
    private SegmentReadState segmentReadState;
    private QuantizationParams quantizationParams;
    private String field;
    private String cacheKey;
}
