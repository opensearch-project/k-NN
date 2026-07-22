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
 *
 * <p>The {@code bits} field must be set explicitly by callers — there is no default. The supported
 * MOS bit widths are {@code 1}, {@code 2}, and {@code 4}; {@link #EMPTY} carries {@code bits=0} as
 * a sentinel. Callers that treat the value as a doc bit width should validate it through
 * {@link FaissSQEncoder#isMosBits(int)}.
 */
@Builder
@Getter
@EqualsAndHashCode
public class SQConfig {
    private int bits;

    public static final SQConfig EMPTY = SQConfig.builder().bits(0).build();
}
