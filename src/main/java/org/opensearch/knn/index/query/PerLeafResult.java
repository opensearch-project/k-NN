/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.Setter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;

/**
 * Represents the per-segment (leaf-level) result of a vector search operation.
 * <p>
 * This class encapsulates the intermediate search state and results produced from
 * a single {@link LeafReaderContext} during approximate or exact vector search.
 * It stores the active filter bitset (if any), its cardinality, the top document
 * results for that segment, and the search mode used.
 * <p>
 * Instances of this class are typically aggregated at a higher level to produce
 * the global {@code TopDocs} result set.
 */
@Getter
public class PerLeafResult {
    /**
     * An immutable, empty {@link BitSet} implementation used to represent
     * the absence of filter bits without incurring null checks or allocations.
     */
    public static final BitSet MATCH_ALL_BIT_SET = new BitSet() {
        @Override
        public void set(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean getAndSet(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void clear(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void clear(int startIndex, int endIndex) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int cardinality() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int approximateCardinality() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int prevSetBit(int index) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int nextSetBit(int start, int end) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long ramBytesUsed() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean get(int i) {
            return true;
        }

        @Override
        public int length() {
            throw new UnsupportedOperationException();
        }
    };

    /**
     * An immutable, empty {@link BitSet} implementation used to represent
     * the absence of filter bits without incurring null checks or allocations.
     */
    private static final BitSet MATCH_NO_BIT_SET = new BitSet() {
        @Override
        public void set(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean getAndSet(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void clear(int i) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void clear(int startIndex, int endIndex) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int cardinality() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int approximateCardinality() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int prevSetBit(int index) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int nextSetBit(int start, int end) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long ramBytesUsed() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean get(int i) {
            return false;
        }

        @Override
        public int length() {
            throw new UnsupportedOperationException();
        }
    };

    // A statically defined empty {@code PerLeafResult} used as a lightweight placeholder when a segment produces no hits.
    public static final PerLeafResult EMPTY_RESULT = new PerLeafResult(
        MATCH_NO_BIT_SET,
        0,
        TopDocsCollector.EMPTY_TOPDOCS,
        SearchMode.EXACT_SEARCH
    );

    /**
     * Indicates the search mode applied within a segment. Either exact or approximate nearest neighbor (ANN) search.
     */
    public enum SearchMode {
        EXACT_SEARCH,
        APPROXIMATE_SEARCH,
    }

    // Active filter bitset limiting document candidates in this leaf (may be empty).
    @Nullable
    private final BitSet filterBits;

    // Cardinality of {@link #filterBits}, used for filtering optimizations.
    private final int filterBitsCardinality;

    // Top document results for this leaf segment.
    @Setter
    private TopDocs result;

    // Indicates whether this result was produced via exact or approximate search.
    private final SearchMode searchMode;

    /**
     * Constructs a new {@code PerLeafResult}.
     *
     * @param filterBits the document filter bitset for this leaf, or {@code null} if none
     * @param filterBitsCardinality the number of bits set in {@code filterBits}
     * @param result the top document results for this leaf
     * @param searchMode the search mode (exact or approximate) used
     */
    public PerLeafResult(final BitSet filterBits, final int filterBitsCardinality, final TopDocs result, final SearchMode searchMode) {
        this.filterBits = filterBits == null ? MATCH_ALL_BIT_SET : filterBits;
        this.filterBitsCardinality = filterBitsCardinality;
        this.result = result;
        this.searchMode = searchMode;
    }
}
