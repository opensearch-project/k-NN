/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.luceneonfaiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.SpaceType;

import java.io.IOException;

import static org.opensearch.knn.index.codec.luceneonfaiss.FaissHNSWFlatIndex.IHNF;
import static org.opensearch.knn.index.codec.luceneonfaiss.FaissIdMapIndex.IXMP;
import static org.opensearch.knn.index.codec.luceneonfaiss.FaissIndexFlat.IXF2;
import static org.opensearch.knn.index.codec.luceneonfaiss.FaissIndexFlat.IXFI;

@Getter
public abstract class FaissIndex {
    // Vector dimension
    private int dimension;
    // Total number of vectors saved within this index.
    private long totalNumberOfVectors;
    // Space type used to index vectors in this index.
    private SpaceType spaceType;

    public static FaissIndex load(IndexInput input) throws IOException {
        final String indexName = readFourBytes(input);

        switch (indexName) {
            case IXMP: {
                return FaissIdMapIndex.load(input);
            }
            case IHNF: {
                return FaissHNSWFlatIndex.load(input);
            }
            case IXF2:
                // Fallthrough
            case IXFI:
                return FaissIndexFlat.load(input, indexName);
            default: {
                throw new IllegalStateException("Partial loading does not support [" + indexName + "].");
            }
        }
    }

    static protected void readCommonHeader(IndexInput readStream, FaissIndex index) throws IOException {
        index.dimension = readStream.readInt();
        index.totalNumberOfVectors = readStream.readLong();
        // consume 2 dummy deprecated fields.
        readStream.readLong();
        readStream.readLong();

        // We don't use this field
        final boolean isTrained = readStream.readByte() == 1;

        final int metricTypeIndex = readStream.readInt();
        if (metricTypeIndex > 1) {
            throw new IllegalStateException("Partial loading does not support metric type index=[" + metricTypeIndex + "] from FAISS.");
        }

        if (metricTypeIndex == 0) {
            index.spaceType = SpaceType.INNER_PRODUCT;
        } else if (metricTypeIndex == 1) {
            index.spaceType = SpaceType.L2;
        } else {
            throw new IllegalStateException("Partial loading does not support metric type index=" + metricTypeIndex + " from FAISS.");
        }
    }

    static private String readFourBytes(IndexInput input) throws IOException {
        final byte[] fourBytes = new byte[4];
        input.readBytes(fourBytes, 0, fourBytes.length);
        return new String(fourBytes);
    }

    /**
     * Returns a unique signature of the FAISS index.
     *
     * @return Index type string.
     */
    public abstract String getIndexType();
}
