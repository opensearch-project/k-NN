/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.util.Bits;
import org.opensearch.common.Nullable;

import java.util.Collections;
import java.util.Map;

@Getter
public class PerLeafResult {
    public static final PerLeafResult EMPTY_RESULT = new PerLeafResult(new Bits.MatchNoBits(0), Collections.emptyMap());
    @Nullable
    private final Bits filterBits;
    @Setter
    private Map<Integer, Float> result;

    public PerLeafResult(final Bits filterBits, final Map<Integer, Float> result) {
        this.filterBits = filterBits == null ? new Bits.MatchAllBits(0) : filterBits;
        this.result = result;
    }
}
