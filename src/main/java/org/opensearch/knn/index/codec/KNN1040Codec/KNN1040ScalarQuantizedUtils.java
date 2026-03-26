/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Locale;

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
 * ({@link KNN1040ScalarQuantizedVectorScorer}) to obtain quantized vectors for SIMD-accelerated scoring.
 */
@UtilityClass
class KNN1040ScalarQuantizedUtils {
    private static final String QUANTIZED_VECTOR_VALUES_FIELD_NAME = "quantizedVectorValues";

    /**
     * Extracts {@link QuantizedByteVectorValues} from the given {@link KnnVectorValues} via reflection.
     *
     * <p>The {@code floatVectorValues} parameter is expected to be a {@code ScalarQuantizedVectorValues}
     * instance returned by {@code Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues()}.
     * This wrapper holds a private {@code quantizedVectorValues} field containing the 1-bit binary
     * quantized codes and their correction factors (lower/upper intervals, additional correction,
     * and quantized component sum).
     *
     * @param floatVectorValues the vector values instance to extract quantized values from;
     *                          typically a {@code ScalarQuantizedVectorValues}
     * @return the extracted {@link QuantizedByteVectorValues}, or {@code null} if not found
     * and {@code throwExceptionIfNotFound} is {@code false}
     * @throws IOException if extraction fails and {@code throwExceptionIfNotFound} is {@code true}
     */
    /**
     * Normalizes the vector in place if the similarity function is COSINE and the vector is not already a unit vector.
     * FAISS already normalizes query vectors for cosine, so this avoids redundant work while ensuring
     * Lucene's {@link org.apache.lucene.util.quantization.OptimizedScalarQuantizer} assertion is satisfied.
     *
     * @param vector the vector to conditionally normalize
     * @param similarityFunction the similarity function in use
     */
    public static void normalizeIfNeeded(final float[] vector, final VectorSimilarityFunction similarityFunction) {
        if (similarityFunction == VectorSimilarityFunction.COSINE && !VectorUtil.isUnitVector(vector)) {
            VectorUtil.l2normalize(vector);
        }
    }

    public static QuantizedByteVectorValues extractQuantizedByteVectorValues(final KnnVectorValues floatVectorValues) throws IOException {
        try {
            final Field f = floatVectorValues.getClass().getDeclaredField(QUANTIZED_VECTOR_VALUES_FIELD_NAME);
            f.setAccessible(true);
            return (QuantizedByteVectorValues) f.get(floatVectorValues);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "Failed to extract QuantizedByteVectorValues from floatVectorValues [%s/%s]."
                        + " This may indicate an incompatible Lucene version.",
                    floatVectorValues.getClass().getSimpleName(),
                    floatVectorValues
                ),
                e
            );
        }
    }
}
