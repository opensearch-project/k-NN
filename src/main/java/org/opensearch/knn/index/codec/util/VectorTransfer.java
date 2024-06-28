/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import lombok.Data;
import org.opensearch.knn.index.KNNSettings;

import java.io.ByteArrayInputStream;

@Data
public abstract class VectorTransfer {
    protected final long vectorsStreamingMemoryLimit;
    protected long totalLiveDocs;
    protected long vectorsPerTransfer;
    protected long vectorAddress;
    protected int dimension;

    public VectorTransfer() {
        this.vectorsStreamingMemoryLimit = KNNSettings.getVectorStreamingMemoryLimit().getBytes();
        this.vectorsPerTransfer = Integer.MIN_VALUE;
    }

    abstract public void init(final long totalLiveDocs);

    abstract public void addVector(final ByteArrayInputStream byteStream);

    abstract public void flush();

    abstract public SerializationMode getSerializationMode(final ByteArrayInputStream byteStream);
}
