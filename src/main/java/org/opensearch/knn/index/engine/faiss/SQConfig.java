/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * Configuration for the SQ (Scalar Quantization) encoder stored as a field attribute.
 */
@Builder
@Getter
@EqualsAndHashCode
public class SQConfig {
    @Builder.Default
    private int bits = FaissSQEncoder.Bits.ONE.getValue();

    public static final SQConfig EMPTY = SQConfig.builder().bits(0).build();
}
