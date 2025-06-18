/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.cagra;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.faiss.AbstractFaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

/**
 * This class represents FAISS CAGRA index which is built with GPU.
 * The index has the same structure with a plain HNSW index built with CPU.
 * But in addition to HNSW graph, it contains special CAGRA meta info such as `numBaseLevelSearchEntryPoint`.
 */
public class FaissHNSWCagraIndex extends AbstractFaissHNSWIndex {
    public static final String IHNC = "IHNc";

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexHHNSWCagra to create a full base layer graph that is
    // used when GpuIndexCagra::copyFrom(IndexHNSWCagra*) is invoked.
    // See https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h#L47
    private boolean keepMaxSizeLevel0;

    // When set to true, the index is immutable.
    // This option is used to copy the knn graph from GpuIndexCagra
    // to the base level of IndexHNSWCagra without adding upper levels.
    // Doing so enables to search the HNSW index, but removes the
    // ability to add vectors.
    // See https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h#L180
    private boolean baseLevelOnly;

    // When `base_level_only` is set to `True`, the search function
    // searches only the base level knn graph of the HNSW index.
    // This parameter selects the entry point by randomly selecting
    // some points and using the best one.
    // See https://github.com/facebookresearch/faiss/blob/main/faiss/IndexHNSW.h#L186
    private int numBaseLevelSearchEntryPoint;

    public FaissHNSWCagraIndex() {
        super(IHNC, new FaissCagraHNSW());
    }

    /**
     * Loading CAGRA HNSW graph and nested storage index.
     * For more details, please refer to
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L967">...</a>
     * @param input Input stream to FAISS index file.
     * @throws IOException
     */
    @Override
    protected void doLoad(final IndexInput input) throws IOException {
        // Read common header
        readCommonHeader(input);

        // Read CAGRA meta info
        keepMaxSizeLevel0 = input.readByte() != 0;
        baseLevelOnly = input.readByte() != 0;
        numBaseLevelSearchEntryPoint = input.readInt();

        // Partial load HNSW graph
        faissHnsw.load(input, getTotalNumberOfVectors());
        ((FaissCagraHNSW) faissHnsw).setNumBaseLevelSearchEntryPoints(numBaseLevelSearchEntryPoint);

        // Partial load flat vector storage
        flatVectors = FaissIndex.load(input);
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return flatVectors.getVectorEncoding();
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        return flatVectors.getFloatValues(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        return flatVectors.getByteValues(indexInput);
    }
}
