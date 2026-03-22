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

/**
 * Utility class for extracting quantized vector values from Lucene's internal reader structures.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * from {@code getFloatVectorValues()}, which wraps both raw float vectors and their quantized counterparts.
 * The quantized values are stored in a private {@code quantizedVectorValues} field and are not exposed
 * through any public API. This utility uses reflection to access that field.
 *
 * <p>This is used by both the write path ({@link Faiss1040ScalarQuantizedKnnVectorsWriter}) to extract
 * quantized vectors for native HNSW graph construction, and the search path
 * ({@link Faiss104ScalarQuantizedVectorScorer}) to obtain quantized vectors for SIMD-accelerated scoring.
 */
@UtilityClass
class Faiss1040ScalarQuantizedUtils {

    /**
     * Extracts {@link QuantizedByteVectorValues} from the given {@link KnnVectorValues} via reflection.
     *
     * <p>The {@code floatVectorValues} parameter is expected to be a {@code ScalarQuantizedVectorValues}
     * instance returned by {@code Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues()}.
     * This wrapper holds a private {@code quantizedVectorValues} field containing the 1-bit binary
     * quantized codes and their correction factors (lower/upper intervals, additional correction,
     * and quantized component sum).
     *
     * @param floatVectorValues    the vector values instance to extract quantized values from;
     *                             typically a {@code ScalarQuantizedVectorValues}
     * @param throwExceptionIfNotFound if {@code true}, throws {@link IOException} when the field
     *                                 cannot be found (e.g., incompatible Lucene version);
     *                                 if {@code false}, returns {@code null} silently
     * @return the extracted {@link QuantizedByteVectorValues}, or {@code null} if not found
     *         and {@code throwExceptionIfNotFound} is {@code false}
     * @throws IOException if extraction fails and {@code throwExceptionIfNotFound} is {@code true}
     */
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
                    "Failed to extract QuantizedByteVectorValues from floatVectorValues ["
                        + floatVectorValues.getClass().getSimpleName()
                        + "/"
                        + floatVectorValues
                        + "]"
                        + " This may indicate an incompatible Lucene "
                        + "version.",
                    e
                );
            }
        }

        return null;
    }
}
