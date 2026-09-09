/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.index.VectorDataType;

/**
 * SQ 1-bit format for {@code half_float} with the same mechanism as {@link KNN1040ScalarQuantizedVectorsFormat},
 * but with an FP16 raw delegate instead of FP32. Separate class as Lucene reconstructs formats
 * by name via a no-arg constructor, so sharing a name with the FLOAT variant would silently pick the
 * wrong delegate on read.
 */
public class KNN1040HalfFloatScalarQuantizedVectorsFormat extends KNN1040ScalarQuantizedVectorsFormat {

    public KNN1040HalfFloatScalarQuantizedVectorsFormat() {
        this(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);
    }

    public KNN1040HalfFloatScalarQuantizedVectorsFormat(final ScalarEncoding encoding) {
        super(encoding, VectorDataType.HALF_FLOAT);
    }
}
