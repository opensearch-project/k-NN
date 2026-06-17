/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;

import java.io.IOException;

/**
 * Utility methods for nested kNN vector queries.
 */
final class NestedKnnUtil {

    private NestedKnnUtil() {}

    /**
     * Checks if a segment has no parent documents (no nested objects).
     * This is used to short-circuit kNN search on segments where no vectors exist,
     * preventing a NPE in Lucene's TimeLimitingKnnCollectorManager which wraps a null
     * collector when DiversifyingNearestChildrenKnnCollectorManager returns null.
     *
     * @param parentFilter the parent document filter
     * @param context the leaf reader context for the segment
     * @return true if the segment has no parent documents
     */
    static boolean hasNoParentDocs(BitSetProducer parentFilter, LeafReaderContext context) throws IOException {
        BitSet parentBitSet = parentFilter.getBitSet(context);
        return parentBitSet == null;
    }
}
