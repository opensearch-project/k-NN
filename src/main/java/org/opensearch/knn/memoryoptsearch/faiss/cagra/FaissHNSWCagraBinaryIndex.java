/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.cagra;

import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.FlatVectorsReaderWithFieldName;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexLoadUtils;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;

import java.io.IOException;

/**
 * This class represents FAISS CAGRA index which is built with GPU.
 * The index has the same structure with a plain HNSW index built with CPU.
 * But in addition to HNSW graph, it contains special CAGRA meta info such as `numBaseLevelSearchEntryPoint`.
 */
public class FaissHNSWCagraBinaryIndex extends FaissBinaryHnswIndex {
    public static final String IBHC = "IBHc";

    private boolean keepMaxSizeLevel0;

    private boolean baseLevelOnly;

    public FaissHNSWCagraBinaryIndex() {
        super(IBHC, new FaissCagraHNSW());
    }

    @Override
    protected void doLoad(final IndexInput input, FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName) throws IOException {
        // Read common binary index header
        readBinaryCommonHeader(input);

        keepMaxSizeLevel0 = input.readByte() == 1;
        baseLevelOnly = input.readByte() == 1;
        final int numBaseLevelSearchEntryPoint = input.readInt();
        ((FaissCagraHNSW) faissHnsw).setNumBaseLevelSearchEntryPoints(numBaseLevelSearchEntryPoint);

        faissHnsw.load(input, getTotalNumberOfVectors());
        storage = FaissIndexLoadUtils.toBinaryIndex(FaissIndex.load(input, flatVectorsReaderWithFieldName));
    }
}
