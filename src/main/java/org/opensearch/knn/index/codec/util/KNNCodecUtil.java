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
    // References to objects are 4 bytes in size
    public static final int JAVA_REFERENCE_SIZE = 4;
    // Each array in Java has a header that is 12 bytes
    public static final int JAVA_ARRAY_HEADER_SIZE = 12;
    // Java rounds each array size up to multiples of 8 bytes
    public static final int JAVA_ROUNDING_NUMBER = 8;

    @AllArgsConstructor
    public static final class VectorBatch {
        public int[] docs;
        @Getter
        @Setter
        private long vectorAddress;
        @Getter
        @Setter
        private int dimension;
        public SerializationMode serializationMode;
        public boolean finished;
    }

    /**
     * Extract docIds and vectors from binary doc values.
    *
    * @param values Binary doc values
    * @param vectorTransfer Utility to make transfer
    * @return KNNCodecUtil.Pair representing doc ids and corresponding vectors
    * @throws IOException thrown when unable to get binary of vectors
    */
    public static KNNCodecUtil.VectorBatch getVectorBatch(
        final BinaryDocValues values,
        final VectorTransfer vectorTransfer,
        boolean iterative
    ) throws IOException {
        List<Integer> docIdList = new ArrayList<>();
        SerializationMode serializationMode = SerializationMode.COLLECTION_OF_FLOATS;
        if (iterative) {
            // Initializing with a value of zero means to only allocate as much memory on JNI as
            // we have inserted for vectors in java side
            vectorTransfer.init(0);
        } else {
            vectorTransfer.init(getTotalLiveDocsCount(values));
        }
        int doc = values.docID();
        // THIS IS A HACK. We check the first document before calling this function in KNNIndexBuilder.
        if(doc != 0) {
            doc = values.nextDoc();
        }
        for (; doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            serializationMode = vectorTransfer.getSerializationMode(bytesref);
            vectorTransfer.transfer(bytesref);
            docIdList.add(doc);
            // Semi-hacky way to check if the streaming limit has been reached
            if (iterative && vectorTransfer.getVectorAddress() != 0) {
                break;
            }
        }
        vectorTransfer.close();

        boolean finished = doc == DocIdSetIterator.NO_MORE_DOCS;

        return new KNNCodecUtil.VectorBatch(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorTransfer.getVectorAddress(),
            vectorTransfer.getDimension(),
            serializationMode,
            finished
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
