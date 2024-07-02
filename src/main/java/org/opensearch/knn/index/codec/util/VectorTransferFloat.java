/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.opensearch.knn.jni.JNICommons;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;

public class VectorTransferFloat extends VectorTransfer {
    private List<float[]> vectorList;

    public VectorTransferFloat() {
        super();
        vectorList = new ArrayList<>();
    }

    @Override
    public void init(final long totalLiveDocs) {
        this.totalLiveDocs = totalLiveDocs;
        vectorList.clear();
    }

    @Override
    public void addVector(final ByteArrayInputStream byteStream) {
        final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
        final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
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
            flush();
        }
    }

    @Override
    public void flush() {
        vectorAddress = JNICommons.storeVectorData(vectorAddress, vectorList.toArray(new float[][] {}), totalLiveDocs * dimension);
        vectorList.clear();
    }

    @Override
    public SerializationMode getSerializationMode(final ByteArrayInputStream byteStream) {
        return KNNVectorSerializerFactory.serializerModeFromStream(byteStream);
    }
}
