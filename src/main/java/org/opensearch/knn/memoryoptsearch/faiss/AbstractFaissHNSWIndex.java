/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;

public abstract class AbstractFaissHNSWIndex extends FaissIndex {
    @Getter
    protected FaissHNSW hnsw = new FaissHNSW();
    protected FaissIndex flatVectors;

    public AbstractFaissHNSWIndex(final String indexType, final FaissHNSW hnsw) {
        super(indexType);
        this.hnsw = hnsw;
    }
}
