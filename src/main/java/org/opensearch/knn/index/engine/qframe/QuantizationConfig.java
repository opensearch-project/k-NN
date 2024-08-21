/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.qframe;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

/**
 * Configuration for quantization
 */
@Builder
@Getter
@EqualsAndHashCode
public class QuantizationConfig {
    @Builder.Default
    private ScalarQuantizationType quantizationType = null;
    public static final QuantizationConfig EMPTY = QuantizationConfig.builder().build();
}
