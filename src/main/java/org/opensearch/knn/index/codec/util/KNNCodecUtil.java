/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;
    // References to objects are 8 bytes in size
    public static final int JAVA_REFERENCE_SIZE = 8;
    // Each array in Java has a header that is 12 bytes
    public static final int JAVA_ARRAY_HEADER_SIZE = 12;
    // Java rounds each array size up to multiples of 8 bytes
    public static final int JAVA_ROUNDING_NUMBER = 8;

    @AllArgsConstructor
    public static final class Pair {
        public int[] docs;
        @Getter
        @Setter
        private long vectorAddress;
        @Getter
        @Setter
        private int dimension;
        public SerializationMode serializationMode;
    }

    /**
     * Extract docIds and vectors from binary doc values.
     *
     * @param values Binary doc values
     * @param vectorTransfer Utility to make transfer
     * @return KNNCodecUtil.Pair representing doc ids and corresponding vectors
     * @throws IOException thrown when unable to get binary of vectors
     */
    public static KNNCodecUtil.Pair getPair(final BinaryDocValues values, final VectorTransfer vectorTransfer) throws IOException {
        List<Integer> docIdList = new ArrayList<>();
        SerializationMode serializationMode = SerializationMode.COLLECTION_OF_FLOATS;
        vectorTransfer.init(getTotalLiveDocsCount(values));
        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            serializationMode = vectorTransfer.getSerializationMode(bytesref);
            vectorTransfer.transfer(bytesref);
            docIdList.add(doc);
        }
        vectorTransfer.close();
        return new KNNCodecUtil.Pair(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorTransfer.getVectorAddress(),
            vectorTransfer.getDimension(),
            serializationMode
        );
    }

    /**
     * This method provides a rough estimate of the number of bytes used for storing an array with the given parameters.
     * @param numVectors number of vectors in the array
     * @param vectorLength the length of each vector
     * @param serializationMode serialization mode
     * @return rough estimate of number of bytes used to store an array with the given parameters
     */
    public static long calculateArraySize(int numVectors, int vectorLength, SerializationMode serializationMode) {
        // For more information about array storage in Java, visit https://www.javamex.com/tutorials/memory/array_memory_usage.shtml
        // Note: java reference size is 8 bytes for 64 bit machines and 4 bytes for 32 bit machines, this method assumes 64 bit
        if (serializationMode == SerializationMode.ARRAY) {
            int sizeOfVector = vectorLength * FLOAT_BYTE_SIZE + JAVA_ARRAY_HEADER_SIZE;
            int sizeOfVectorArray = roundVectorSize(sizeOfVector) * numVectors;
            int sizeOfReferenceArray = numVectors * JAVA_REFERENCE_SIZE + JAVA_ARRAY_HEADER_SIZE;
            sizeOfReferenceArray = roundVectorSize(sizeOfReferenceArray);
            return sizeOfReferenceArray + sizeOfVectorArray;
        } else if (serializationMode == SerializationMode.COLLECTION_OF_FLOATS) {
            int sizeOfVector = vectorLength * FLOAT_BYTE_SIZE;
            int sizeOfVectorArray = roundVectorSize(sizeOfVector) * numVectors;
            int sizeOfReferenceArray = numVectors * JAVA_REFERENCE_SIZE;
            sizeOfReferenceArray = roundVectorSize(sizeOfReferenceArray);
            return sizeOfReferenceArray + sizeOfVectorArray;
        } else if (serializationMode == SerializationMode.COLLECTIONS_OF_BYTES) {
            int sizeOfVector = vectorLength;
            int sizeOfVectorArray = roundVectorSize(sizeOfVector) * numVectors;
            int sizeOfReferenceArray = numVectors * JAVA_REFERENCE_SIZE;
            sizeOfReferenceArray = roundVectorSize(sizeOfReferenceArray);
            return sizeOfReferenceArray + sizeOfVectorArray;
        } else {
            throw new IllegalStateException("Unreachable code");
        }
    }

    private static int roundVectorSize(int vectorSize) {
        if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
            return vectorSize + (JAVA_ROUNDING_NUMBER - vectorSize % JAVA_ROUNDING_NUMBER);
        }
        return vectorSize;
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

    private static long getTotalLiveDocsCount(final BinaryDocValues binaryDocValues) {
        long totalLiveDocs;
        if (binaryDocValues instanceof KNN80BinaryDocValues) {
            totalLiveDocs = ((KNN80BinaryDocValues) binaryDocValues).getTotalLiveDocs();
        } else {
            totalLiveDocs = binaryDocValues.cost();
        }
        return totalLiveDocs;
    }
}
