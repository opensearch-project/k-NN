/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

/**
 * IndexBuildSetup encapsulates the configuration and parameters required for building an index.
 * This includes the size of each vector, the dimensions of the vectors, and any quantization-related
 * settings such as the output and state of quantization.
 */
@Getter
@AllArgsConstructor
public final class IndexBuildSetup {
    /**
     * The number of bytes per vector.
     */
    private final int bytesPerVector;

    /**
     * Dimension of Vector for Indexing
     */
    private final int dimensions;

    /**
     * The quantization output that will hold the quantized vector.
     */
    private final QuantizationOutput quantizationOutput;

    /**
     * The state of quantization, which may include parameters and trained models.
     */
    private final QuantizationState quantizationState;
}
