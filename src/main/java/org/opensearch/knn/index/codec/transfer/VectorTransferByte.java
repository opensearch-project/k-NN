/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.util.ArrayList;
import java.util.List;

/**
 * Vector transfer for byte
 */
public class VectorTransferByte extends VectorTransfer {
    private List<byte[]> vectorList;

    public VectorTransferByte(final long vectorsStreamingMemoryLimit) {
        super(vectorsStreamingMemoryLimit);
        vectorList = new ArrayList<>();
    }

    @Override
    public void init(final long totalLiveDocs) {
        this.totalLiveDocs = totalLiveDocs;
        vectorList.clear();
    }

    @Override
    public void transfer(final BytesRef bytesRef) {
        dimension = bytesRef.length * 8;
        if (vectorsPerTransfer == Integer.MIN_VALUE) {
            vectorsPerTransfer = (bytesRef.length * totalLiveDocs) / vectorsStreamingMemoryLimit;
            // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
            // Doing this will reduce 1 extra trip to JNI layer.
            if (vectorsPerTransfer == 0) {
                vectorsPerTransfer = totalLiveDocs;
            }
        }

        vectorList.add(ArrayUtil.copyOfSubArray(bytesRef.bytes, bytesRef.offset, bytesRef.offset + bytesRef.length));
        if (vectorList.size() == vectorsPerTransfer) {
            transfer();
        }
    }

    @Override
    public void close() {
        transfer();
    }

    @Override
    public SerializationMode getSerializationMode(final BytesRef bytesRef) {
        return SerializationMode.COLLECTIONS_OF_BYTES;
    }

    private void transfer() {
        int lengthOfVector = dimension / 8;
        vectorAddress = JNICommons.storeByteVectorData(vectorAddress, vectorList.toArray(new byte[][] {}), totalLiveDocs * lengthOfVector);
        vectorList.clear();
    }
}
