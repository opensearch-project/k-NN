/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reconstruct;

import lombok.experimental.UtilityClass;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;

@UtilityClass
public class FaissQuantizedValueReconstructorFactory {
    public static FaissQuantizedValueReconstructor create(
        final FaissQuantizerType quantizerType,
        final int dimension,
        final int numOneVectorBits
    ) {

        if (quantizerType == FaissQuantizerType.QT_8BIT_DIRECT_SIGNED) {
            return new Faiss8BitsDirectSignedReconstructor(dimension, numOneVectorBits);
        }
        if (quantizerType == FaissQuantizerType.QT_FP16) {
            return new FaissFP16Reconstructor(dimension, numOneVectorBits);
        }

        throw new UnsupportedFaissIndexException("Unsupported quantizer type: " + quantizerType);
    }
}
