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
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80BinaryDocValues;
import org.opensearch.knn.index.codec.transfer.VectorTransfer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNCodecUtil {
    // Floats are 4 bytes in size
    public static final int FLOAT_BYTE_SIZE = 4;

    @AllArgsConstructor
    public static final class VectorBatch {
        public int[] docs;
        @Getter
        @Setter
        private long vectorAddress;
        @Getter
        @Setter
        private int dimension;
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
        if (iterative) {
            // Initializing with a value of zero means to only allocate as much memory on JNI as
            // we have inserted for vectors in java side
            vectorTransfer.init(0);
        } else {
            vectorTransfer.init(getTotalLiveDocsCount(values));
        }
        for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
            BytesRef bytesref = values.binaryValue();
            vectorTransfer.transfer(bytesref);
            docIdList.add(doc);
            // Semi-hacky way to check if the streaming limit has been reached
            if (iterative && vectorTransfer.numPendingDocs() == 0) {
                break;
            }
        }
        vectorTransfer.close();

        boolean finished = values.docID() == DocIdSetIterator.NO_MORE_DOCS;

        return new KNNCodecUtil.VectorBatch(
            docIdList.stream().mapToInt(Integer::intValue).toArray(),
            vectorTransfer.getVectorAddress(),
            vectorTransfer.getDimension(),
            finished
        );
    }

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
