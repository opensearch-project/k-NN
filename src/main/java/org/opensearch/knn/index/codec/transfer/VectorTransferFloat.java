/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;
import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.util.ArrayList;
import java.util.List;

/**
 * Vector transfer for float
 */
public class VectorTransferFloat extends VectorTransfer {
    private List<float[]> vectorList;

    public VectorTransferFloat(final long vectorsStreamingMemoryLimit) {
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
        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(bytesRef);
        final float[] vector = vectorSerializer.byteToFloatArray(bytesRef);
        dimension = vector.length;

        if (vectorsPerTransfer == Integer.MIN_VALUE) {
            // if vectorsStreamingMemoryLimit is 100 bytes and we have 50 vectors with 5 dimension, then per
            // transfer we have to send 100/(5 * 4) => 5 vectors.
            vectorsPerTransfer = vectorsStreamingMemoryLimit / ((long) dimension * Float.BYTES);
            // If vectorsPerTransfer comes out to be 0, then we set number of vectors per transfer to 1, to ensure that
            // we are sending minimum number of vectors.
            if (vectorsPerTransfer == 0) {
                vectorsPerTransfer = 1;
            }
        }

        vectorList.add(vector);
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
        return KNNVectorSerializerFactory.getSerializerModeFromBytesRef(bytesRef);
    }

    private void transfer() {
        vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        vectorList.clear();
    }
}
