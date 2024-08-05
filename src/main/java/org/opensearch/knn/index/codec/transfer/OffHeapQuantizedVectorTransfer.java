/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.Getter;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.common.KNNVectorUtil.intOverflowSafeArrayList;

/**
 * <p>
 * The class is intended to transfer {@link KNNVectorValues} to off heap memory.
 * It also provides and ability to quantize the vector before it is transferred to offHeap memory.
 * The ability to quantize is added as to not iterate KNN {@link KNNVectorValues} multiple times.
 * </p>
 *
 * <p>
 * The class is not thread safe.
 * </p>
 *
 * @param <T>  byte[] or float[]
 * @param <V>  byte[] or float[]
 */
abstract class OffHeapQuantizedVectorTransfer<T, V> implements VectorTransfer {

    protected static final int DEFAULT_COMPRESSION_FACTOR = 1;

    @Getter
    private long vectorAddress;
    @Getter
    private int[] transferredDocsIds;
    private final int transferLimit;

    // Keeping this as a member variable as this should not be changed considering the vector address is reused between batches
    protected long batchSize;

    private final List<V> vectorsToTransfer;
    private final List<Integer> transferredDocIdsList;

    private final KNNVectorValues<T> vectorValues;

    // TODO: Replace with actual quantization parameters
    private final BiFunction<T, String, V> quantizer;
    private final String quantizationState;

    public OffHeapQuantizedVectorTransfer(
        final KNNVectorValues<T> vectorValues,
        final Long batchSize,
        final BiFunction<T, String, V> quantizer,
        final String quantizationState,
        final int compressionFactor
    ) {
        assert vectorValues.docId() != -1 : "vectorValues docId must be set, iterate it once for vector transfer to succeed";
        assert vectorValues.docId() != NO_MORE_DOCS : "vectorValues already iterated, Nothing to transfer";

        this.quantizer = quantizer;
        this.quantizationState = quantizationState;
        this.transferLimit = (int) Math.max(
            1,
            (int) KNNSettings.getVectorStreamingMemoryLimit().getBytes() / (vectorValues.bytesPerVector() / compressionFactor)
        );
        this.batchSize = batchSize == null ? transferLimit : batchSize;
        this.vectorsToTransfer = intOverflowSafeArrayList(this.batchSize);
        this.transferredDocIdsList = intOverflowSafeArrayList(this.batchSize);
        this.vectorValues = vectorValues;
        this.vectorAddress = 0; // we can allocate initial memory here, currently storeVectorData takes care of it
    }

    @Override
    public void transferBatch() throws IOException {
        if (vectorValues.docId() == NO_MORE_DOCS) {
            // Throwing instead of returning so there is no way client can go into an infinite loop
            throw new IllegalStateException("No more vectors available to transfer");
        }

        assert vectorsToTransfer.isEmpty() : "Last batch wasn't transferred";
        assert transferredDocIdsList.isEmpty() : "Last batch wasn't transferred";

        int totalDocsTransferred = 0;
        boolean freshBatch = true;

        // TODO: Create non-final QuantizationOutput once here and then reuse the output
        while (vectorValues.docId() != NO_MORE_DOCS && totalDocsTransferred < batchSize) {
            V quantizedVector = quantizer.apply(vectorValues.conditionalCloneVector(), quantizationState);

            transferredDocIdsList.add(vectorValues.docId());
            vectorsToTransfer.add(quantizedVector);
            if (vectorsToTransfer.size() == transferLimit) {
                vectorAddress = transfer(vectorsToTransfer, !freshBatch);
                vectorsToTransfer.clear();
                freshBatch = false;
            }
            vectorValues.nextDoc();
            totalDocsTransferred++;
        }

        // Handle vectorsToTransfer size < transferLimit
        if (!vectorsToTransfer.isEmpty()) {
            vectorAddress = transfer(vectorsToTransfer, !freshBatch);
            vectorsToTransfer.clear();
        }

        this.transferredDocsIds = new int[transferredDocIdsList.size()];
        for (int i = 0; i < transferredDocIdsList.size(); i++) {
            transferredDocsIds[i] = transferredDocIdsList.get(i);
        }
        transferredDocIdsList.clear();
    }

    @Override
    public boolean hasNext() {
        return vectorValues.docId() != NO_MORE_DOCS;
    }

    @Override
    public void close() {
        transferredDocIdsList.clear();
        transferredDocsIds = null;
        vectorAddress = 0;
    }

    protected abstract long transfer(final List<V> vectorsToTransfer, final boolean append) throws IOException;
}
