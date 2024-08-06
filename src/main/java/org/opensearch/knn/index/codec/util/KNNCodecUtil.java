/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import com.google.common.annotations.VisibleForTesting;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.core.common.Strings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;
    // References to objects are 4 bytes in size
    public static final int JAVA_REFERENCE_SIZE = 4;
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

    public static long calculateArraySize(int numVectors, int vectorLength, SerializationMode serializationMode) {
        if (serializationMode == SerializationMode.ARRAY) {
            int vectorSize = vectorLength * FLOAT_BYTE_SIZE + JAVA_ARRAY_HEADER_SIZE;
            if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
            }
            int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE) + JAVA_ARRAY_HEADER_SIZE;
            if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
            }
            return vectorsSize;
        } else if (serializationMode == SerializationMode.COLLECTION_OF_FLOATS) {
            int vectorSize = vectorLength * FLOAT_BYTE_SIZE;
            if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
            }
            int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE);
            if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
            }
            return vectorsSize;
        } else if (serializationMode == SerializationMode.COLLECTIONS_OF_BYTES) {
            int vectorSize = vectorLength;
            if (vectorSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorSize += vectorSize % JAVA_ROUNDING_NUMBER;
            }
            int vectorsSize = numVectors * (vectorSize + JAVA_REFERENCE_SIZE);
            if (vectorsSize % JAVA_ROUNDING_NUMBER != 0) {
                vectorsSize += vectorsSize % JAVA_ROUNDING_NUMBER;
            }
            return vectorsSize;
        } else {
            throw new IllegalStateException("Unreachable code");
        }
    }

    public static String buildEngineFileName(String segmentName, String latestBuildVersion, String fieldName, String extension) {
        return String.format("%s%s%s", buildEngineFilePrefix(segmentName), latestBuildVersion, buildEngineFileSuffix(fieldName, extension));
    }

    public static String buildEngineFilePrefix(String segmentName) {
        return String.format("%s_", segmentName);
    }

    /**
     * Build a suffix of file name with provided field name and custom extension.
     * In case where the given field name contains invalid characters for the file name, it will escape them to '.'.
     * Note that, field name MUST NOT contain '.' in it.
     * <p>
     * Ex:
     * With 'my_vector' as field name and '.hnswc' as an extension -> _0_2011_my_vector.hnswc
     * With 'my vector' (blank) as field name and '.hnswc' as an extension -> _0_2011_my.vector.hnswc
     * With 'my/vector' (it has '/') as field name and '.hnswc' as an extension -> _0_2011_my.vector.hnswc
     *
     * @param fieldName : Vector field name
     * @param extension : File extension.
     * @return File suffix containing a field name that escaped with invalid characters.
     */
    public static String buildEngineFileSuffix(String fieldName, String extension) {
        return String.format("_%s%s", escapeInvalidFileNameCharsInFieldName(fieldName), extension);
    }

    /**
     * In case of compound file, extension would be {engine-extension} + 'c' otherwise just be {engine-extension}
     * Ex: _0_2011_my_vector.hnswc, where engine-extension is 'hnsw'.
     */
    public static String buildCompoundFile(String extension, boolean isCompoundFile) {
        if (isCompoundFile) {
            return extension + KNNConstants.COMPOUND_EXTENSION;
        } else {
            return extension;
        }
    }

    @VisibleForTesting
    static String escapeInvalidFileNameCharsInFieldName(final String fieldName) {
        char[] characters = fieldName.toCharArray();
        for (int i = 0; i < characters.length; ++i) {
            if (Strings.INVALID_FILENAME_CHARS.contains(characters[i])) {
                characters[i] = '.';
            }
        }
        return new String(characters);
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
