/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

/**
 * The ValueQuantizationType enum defines the types of value quantization techniques
 * that can be applied in the KNN.
 */
public enum ValueQuantizationType {
    /**
     * SQ (Scalar Quantization) represents a method where each coordinate of the vector is quantized
     * independently.
     */
    SCALAR
}
