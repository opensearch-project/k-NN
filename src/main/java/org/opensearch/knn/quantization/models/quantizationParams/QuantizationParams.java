/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationParams;

import org.opensearch.knn.quantization.enums.QuantizationType;

import java.io.Serializable;

/**
 * Interface for quantization parameters.
 * This interface defines the basic contract for all quantization parameter types.
 * It provides methods to retrieve the quantization type and a unique type identifier.
 * Implementations of this interface are expected to provide specific configurations
 * for various quantization strategies.
 */
public interface QuantizationParams extends Serializable {

    /**
     * Gets the quantization type associated with the parameters.
     * The quantization type defines the overall strategy or method used
     * for quantization, such as VALUE_QUANTIZATION or SPACE_QUANTIZATION.
     *
     * @return the {@link QuantizationType} indicating the quantization method.
     */
    QuantizationType getQuantizationType();

    /**
     * Provides a unique identifier for the quantization parameters.
     * This identifier is typically a combination of the quantization type
     * and additional specifics, and it serves to distinguish between different
     * configurations or modes of quantization.
     *
     * @return a string representing the unique type identifier.
     */
    String getTypeIdentifier();
}
