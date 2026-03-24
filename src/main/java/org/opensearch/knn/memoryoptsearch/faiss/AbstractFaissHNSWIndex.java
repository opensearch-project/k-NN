/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;

public abstract class AbstractFaissHNSWIndex extends FaissIndex implements FaissHNSWProvider {
    @Getter
    protected FaissHNSW faissHnsw;
    @Getter
    protected FaissIndex flatVectors;

    public AbstractFaissHNSWIndex(final String indexType, final FaissHNSW faissHnsw) {
        super(indexType);
        this.faissHnsw = faissHnsw;
    }
}
