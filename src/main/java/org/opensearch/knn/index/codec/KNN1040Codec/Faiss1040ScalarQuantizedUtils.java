/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;
import java.lang.reflect.Field;

@UtilityClass
class Faiss1040ScalarQuantizedUtils {
    public static QuantizedByteVectorValues extractQuantizedByteVectorValues(
        final KnnVectorValues floatVectorValues,
        final boolean throwExceptionIfNotFound
    ) throws IOException {
        try {
            final Field f = floatVectorValues.getClass().getDeclaredField("quantizedVectorValues");
            f.setAccessible(true);
            return (QuantizedByteVectorValues) f.get(floatVectorValues);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            if (throwExceptionIfNotFound) {
                throw new IOException(
                    "Failed to extract QuantizedByteVectorValues from FlatVectorsReader."
                        + " This may indicate an incompatible Lucene "
                        + "version.",
                    e
                );
            }
        }

        return null;
    }
}
