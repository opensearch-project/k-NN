/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;

@RequiredArgsConstructor
public abstract class WrappedFloatVectorValues extends FloatVectorValues {
    protected final FloatVectorValues nestedVectorValues;

    public static FloatVectorValues getBottomFloatVectorValues(KnnVectorValues knnVectorValues) {
        if ((knnVectorValues instanceof FloatVectorValues) == false) {
            return null;
        }

        FloatVectorValues floatVectorValues = ((FloatVectorValues) knnVectorValues);

        while (floatVectorValues instanceof WrappedFloatVectorValues wrappedFloatVectorValues) {
            floatVectorValues = wrappedFloatVectorValues.nestedVectorValues;
        }

        return floatVectorValues;
    }
}
