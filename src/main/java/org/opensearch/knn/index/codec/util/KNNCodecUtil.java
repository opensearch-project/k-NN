/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.index.BinaryDocValues;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;

public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;

    /**
     * This method provides a rough estimate of the number of bytes used for storing an array with the given parameters.
     * @param numVectors number of vectors in the array
     * @param vectorLength the length of each vector
     * @param vectorDataType type of data stored in each vector
     * @return rough estimate of number of bytes used to store an array with the given parameters
     */
    public static long calculateArraySize(int numVectors, int vectorLength, VectorDataType vectorDataType) {
        if (vectorDataType == VectorDataType.FLOAT) {
            return numVectors * vectorLength * FLOAT_BYTE_SIZE;
        } else if (vectorDataType == VectorDataType.BINARY || vectorDataType == VectorDataType.BYTE) {
            return numVectors * vectorLength;
        } else {
            throw new IllegalArgumentException(
                "Float, binary, and byte are the only supported vector data types for array size calculation."
            );
        }
    }

    public static String buildEngineFileName(String segmentName, String latestBuildVersion, String fieldName, String extension) {
        return String.format("%s%s%s", buildEngineFilePrefix(segmentName), latestBuildVersion, buildEngineFileSuffix(fieldName, extension));
    }

    public static String buildEngineFilePrefix(String segmentName) {
        return String.format("%s_", segmentName);
    }

    public static String buildEngineFileSuffix(String fieldName, String extension) {
        return String.format("_%s%s", fieldName, extension);
    }

    public static long getTotalLiveDocsCount(final BinaryDocValues binaryDocValues) {
        long totalLiveDocs;
        if (binaryDocValues instanceof KNN80BinaryDocValues) {
            totalLiveDocs = ((KNN80BinaryDocValues) binaryDocValues).getTotalLiveDocs();
        } else {
            totalLiveDocs = binaryDocValues.cost();
        }
        return totalLiveDocs;
    }
}
