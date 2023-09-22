/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class KNNCodecUtil {

    public static final String HNSW_EXTENSION = ".hnsw";
    public static final String HNSW_COMPOUND_EXTENSION = ".hnswc";
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;
    // References to objects are 4 bytes in size
    public static final int JAVA_REFERENCE_SIZE = 4;
    // Each array in Java has a header that is 12 bytes
    public static final int JAVA_ARRAY_HEADER_SIZE = 12;
    // Java rounds each array size up to multiples of 8 bytes
    public static final int JAVA_ROUNDING_NUMBER = 8;

    public static final class Pair {
        public Pair(int[] docs, float[][] vectors) {
            this.docs = docs;
            this.vectors = vectors;
        }

        public int[] docs;
        public float[][] vectors;
    }

    public static KNNCodecUtil.Pair getFloats(BinaryDocValues values) throws IOException {
        ArrayList<float[]> vectorList = new ArrayList<>();
        ArrayList<Integer> docIdList = new ArrayList<>();
        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesref.bytes, bytesref.offset, bytesref.length)) {
                final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
                vectorList.add(vector);
            }
            docIdList.add(doc);
        }
        return new KNNCodecUtil.Pair(docIdList.stream().mapToInt(Integer::intValue).toArray(), vectorList.toArray(new float[][] {}));
    }

    public static long calculateArraySize(float[][] vectors) {
        int vectorLength = vectors[0].length;
        int numVectors = vectors.length;
        int vectorSize = vectorLength * FLOAT_BYTE_SIZE + JAVA_ARRAY_HEADER_SIZE;
        if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
            vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
        }
        int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE) + JAVA_ARRAY_HEADER_SIZE;
        if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
            vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
        }
        return vectorsSize;
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
}
