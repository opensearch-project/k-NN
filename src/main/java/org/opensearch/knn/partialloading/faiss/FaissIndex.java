/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.search.DocIdAndDistance;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;

import java.io.IOException;

import static org.opensearch.knn.partialloading.faiss.FaissHNSWFlatIndex.IHNF;
import static org.opensearch.knn.partialloading.faiss.FaissIdMapIndex.IXMP;
import static org.opensearch.knn.partialloading.faiss.FaissIndexFlat.IXF2;
import static org.opensearch.knn.partialloading.faiss.FaissIndexFlat.IXFI;

public abstract class FaissIndex {
    int dimension;
    long totalNumberOfVectors;
    boolean isTrained;
    public SpaceType spaceType;

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

    static void readCommonHeader(IndexInput readStream, FaissIndex index) throws IOException {
        index.dimension = readStream.readInt();
        index.totalNumberOfVectors = readStream.readLong();
        // consume 2 dummy deprecated fields.
        readStream.readLong();
        readStream.readLong();
        index.isTrained = readStream.readByte() == 1;

        final int metricTypeIndex = readStream.readInt();
        if (metricTypeIndex > 1) {
            throw new IllegalStateException("Partial loading does not support metric type index=[" + metricTypeIndex + "] from FAISS.");
        }

        // TODO : magic number
        if (metricTypeIndex == 0) {
            index.spaceType = SpaceType.L2;
        } else if (metricTypeIndex == 1) {
            index.spaceType = SpaceType.INNER_PRODUCT;
        }
    }

    static String readFourBytes(IndexInput input) throws IOException {
        final byte[] fourBytes = new byte[4];
        input.readBytes(fourBytes, 0, fourBytes.length);
        return new String(fourBytes);
    }

    public abstract void searchLeaf(
        IndexInput indexInput, DocIdAndDistance[] results, PartialLoadingSearchParameters searchParameters
    ) throws IOException;

    public abstract String getIndexType();
}
