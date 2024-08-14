/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.jni.JNICommons;

import java.io.IOException;
import java.util.List;

/**
 * Transfer float vectors to off heap memory.
 */
public final class OffHeapFloatVectorTransfer extends OffHeapPreprocessedVectorTransfer<float[], float[]> {

    public OffHeapFloatVectorTransfer(KNNFloatVectorValues vectorValues, Long batchSize) throws IOException {
        super(vectorValues, batchSize);
    }

    public OffHeapFloatVectorTransfer(KNNFloatVectorValues vectorValues) throws IOException {
        this(vectorValues, null);
    }

    @Override
    protected int computeTransferLimit(float[] vector) {
        return (int) this.streamingLimit / (vector.length * 4);
    }

    @Override
    protected long transfer(final List<float[]> vectorsToTransfer, boolean append) throws IOException {
        return JNICommons.storeVectorData(
            getVectorAddress(),
            vectorsToTransfer.toArray(new float[][] {}),
            this.batchSize * vectorsToTransfer.get(0).length,
            append
        );
    }

    @Override
    public void close() {
        super.close();
        JNICommons.freeVectorData(getVectorAddress());
    }
}
