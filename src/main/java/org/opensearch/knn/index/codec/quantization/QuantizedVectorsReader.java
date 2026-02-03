/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.quantization;

import org.apache.lucene.index.ByteVectorValues;

import java.io.IOException;

/**
 * Interface for reading quantized vector values from an index.
 */
public interface QuantizedVectorsReader {

    /**
     * Returns the quantized byte vector values for the specified field.
     *
     * @param fieldName the name of the vector field
     * @return {@link ByteVectorValues} containing the quantized vectors
     * @throws UnsupportedOperationException if quantized vectors are not available for the field
     */
    ByteVectorValues getQuantizedVectorValues(String fieldName) throws IOException;

}
