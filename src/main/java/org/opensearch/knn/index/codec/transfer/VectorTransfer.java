/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.transfer;

import lombok.Data;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.SerializationMode;

/**
 * Abstract class to transfer vector value from Java to native memory
 */
@Data
public abstract class VectorTransfer {
    protected final long vectorsStreamingMemoryLimit;
    protected long totalLiveDocs;
    protected long vectorsPerTransfer;
    protected long vectorAddress;
    protected int dimension;

    public VectorTransfer(final long vectorsStreamingMemoryLimit) {
        this.vectorsStreamingMemoryLimit = vectorsStreamingMemoryLimit;
        this.vectorsPerTransfer = Integer.MIN_VALUE;
    }

    /**
     * Initialize the transfer
     *
     * @param totalLiveDocs total number of vectors to be transferred
     */
    abstract public void init(final long totalLiveDocs);

    /**
     * Transfer a single vector
     *
     * @param bytesRef a vector in bytes format
     */
    abstract public void transfer(final BytesRef bytesRef);

    /**
     * Close the transfer
     */
    abstract public void close();

    /**
     * Get serialization mode of given byte stream
     *
     * @param bytesRef bytes of a vector
     * @return serialization mode
     */
    abstract public SerializationMode getSerializationMode(final BytesRef bytesRef);
}
