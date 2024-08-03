/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.enums;

/**
 * The QuantizationType enum represents the different types of quantization
 * that can be applied in the KNN.
 *
 * <ul>
 *   <li><b>SPACE_QUANTIZATION:</b> This type of quantization focuses on the space
 *       or the representation of the data vectors. It is commonly used for techniques
 *       that reduce the dimensionality or discretize the data space.</li>
 *   <li><b>VALUE_QUANTIZATION:</b> This type of quantization focuses on the values
 *       within the data vectors. It involves mapping continuous values into discrete
 *       values, which can be useful for compressing data or reducing the precision
 *       of the representation.</li>
 * </ul>
 */
public enum QuantizationType {
    /**
     * Represents space quantization, typically involving dimensionality reduction
     * or space partitioning techniques.
     */
    SPACE,

    /**
     * Represents value quantization, typically involving the conversion of continuous
     * values into discrete ones.
     */
    VALUE,
}
