/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.index.codec.util.SerializationMode;
import org.opensearch.knn.jni.JNICommons;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Vector transfer for byte vectors by casting them as float vectors
 */
public class VectorTransferByteToFloat extends VectorTransfer {
    private List<float[]> vectorList;

    public VectorTransferByteToFloat(final long vectorsStreamingMemoryLimit) {
        super(vectorsStreamingMemoryLimit);
        vectorList = new ArrayList<>();
    }

    @Override
    public void init(long totalLiveDocs) {
        this.totalLiveDocs = totalLiveDocs;
        vectorList.clear();
    }

    @Override
    public void transfer(ByteArrayInputStream byteStream) {
        final float[] vector = byteToFloatArray(byteStream);
        dimension = vector.length;

        if (vectorsPerTransfer == Integer.MIN_VALUE) {
            vectorsPerTransfer = (dimension * Float.BYTES * totalLiveDocs) / vectorsStreamingMemoryLimit;
            // This condition comes if vectorsStreamingMemoryLimit is higher than total number floats to transfer
            // Doing this will reduce 1 extra trip to JNI layer.
            if (vectorsPerTransfer == 0) {
                vectorsPerTransfer = totalLiveDocs;
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
    public SerializationMode getSerializationMode(ByteArrayInputStream byteStream) {
        return SerializationMode.COLLECTION_OF_FLOATS;
    }

    private void transfer() {
        vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        vectorList.clear();
    }

    // cast byte vector array into float vector array
    private static float[] byteToFloatArray(ByteArrayInputStream byteStream) {
        final byte[] vectorAsByteArray = byteStream.readAllBytes();
        final int sizeOfFloatArray = vectorAsByteArray.length;
        final float[] vector = new float[sizeOfFloatArray];
        for (int i = 0; i < sizeOfFloatArray; i++) {
            vector[i] = vectorAsByteArray[i];
        }
        return vector;
    }
}
