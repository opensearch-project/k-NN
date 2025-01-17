/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.faiss;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.search.IdAndDistance;
import org.opensearch.knn.partialloading.search.PartialLoadingSearchParameters;

import java.io.IOException;

import static org.opensearch.knn.partialloading.faiss.FaissHNSWFlatIndex.IHNF;
import static org.opensearch.knn.partialloading.faiss.FaissIdMapIndex.IXMP;
import static org.opensearch.knn.partialloading.faiss.FaissIndexFlat.IXF2;
import static org.opensearch.knn.partialloading.faiss.FaissIndexFlat.IXFI;

/**
 * A top-level abstract FAISS index that represents all available index types in FAISS.
 * All ported FAISS indices must inherit from this and implement search functionality to find the nearest vectors to the query vector.
 * The {@link FaissIndex} can also contain nested indices, allowing searches to be delegated, forming a composite index.
 */
@Getter
public abstract class FaissIndex {
    // Vector dimension
    private int dimension;
    // Total number of vectors saved within this index.
    private long totalNumberOfVectors;
    // Space type used to index vectors in this index.
    private SpaceType spaceType;

    public static FaissIndex partiallyLoad(IndexInput input) throws IOException {
        final String indexName = readFourBytes(input);

        switch (indexName) {
            case IXMP: {
                return FaissIdMapIndex.partiallyLoad(input);
            }
            case IHNF: {
                return FaissHNSWFlatIndex.partiallyLoad(input);
            }
            case IXF2:
                // Fallthrough
            case IXFI:
                return FaissIndexFlat.partiallyLoad(input, indexName);
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
            index.spaceType = SpaceType.L2;
        } else if (metricTypeIndex == 1) {
            index.spaceType = SpaceType.INNER_PRODUCT;
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
     * Performs a vector search to find the nearest vectors to the query vector.
     * The client calling this API must provide a non-null `results` array containing non-null instances, whose length is greater than `k`
     * as specified in `searchParameters`. Otherwise, a {@link NullPointerException} will be thrown.
     * The resulting {@link IdAndDistance} will contain a pair of the Lucene document ID and its distance to the query vector.
     *
     * @param indexInput An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @param results A result array containing non-null pairs of vector IDs and their distances. After the search, it is updated by
     *                extracting elements from the result max-heap.
     * @param searchParameters HNSW search parameters, including efSearch, allow customization. If efSearch is provided, it will override
     *                        the default value.
     * @throws IOException
     */
    public abstract void search(
        IndexInput indexInput, IdAndDistance[] results, PartialLoadingSearchParameters searchParameters
    ) throws IOException;

    /**
     * Returns a unique signature of the FAISS index.
     *
     * @return
     */
    public abstract String getIndexType();
}
