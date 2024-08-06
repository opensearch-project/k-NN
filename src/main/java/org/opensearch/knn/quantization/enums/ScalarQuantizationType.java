/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

/**
 * The SQTypes enum defines the various scalar quantization types that can be used
 * in the KNN  for vector quantization.
 * Each type corresponds to a different bit-width representation of the quantized values.
 */
public enum ScalarQuantizationType {
    /**
     * ONE_BIT quantization uses a single bit per coordinate.
     */
    ONE_BIT,

    /**
     * TWO_BIT quantization uses two bits per coordinate.
     */
    TWO_BIT,

    /**
     * FOUR_BIT quantization uses four bits per coordinate.
     */
    FOUR_BIT,

    /**
     * UNSUPPORTED_TYPE is used to denote quantization types that are not supported.
     * This can be used as a placeholder or default value.
     */
    UNSUPPORTED_TYPE
}
