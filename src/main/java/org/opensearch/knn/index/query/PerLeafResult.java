/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.util.Bits;
import org.opensearch.common.Nullable;

@Getter
public class PerLeafResult {
    public static final PerLeafResult EMPTY_RESULT = new PerLeafResult(new Bits.MatchNoBits(0), TopDocsCollector.EMPTY_TOPDOCS);
    @Nullable
    private final Bits filterBits;
    @Setter
    private TopDocs result;

    public PerLeafResult(final Bits filterBits, final TopDocs result) {
        this.filterBits = filterBits == null ? new Bits.MatchAllBits(0) : filterBits;
        this.result = result;
    }
}
