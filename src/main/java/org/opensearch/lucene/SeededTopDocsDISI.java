/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import java.io.IOException;
import java.util.Arrays;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.TopDocs;

/**
 * A {@link DocIdSetIterator} implementation that iterates over the document IDs
 * collected in a {@link TopDocs} object.
 * <p>
 * This class extracts document IDs from the given {@link TopDocs}, sorts them
 * in ascending order, and then provides sequential access to those IDs.
 * <p>
 * It can be used to re-iterate over the documents returned by a collector,
 * ensuring deterministic iteration order regardless of the original collection sequence.
 */
public class SeededTopDocsDISI extends DocIdSetIterator {

    /** Sorted array of document IDs extracted from {@link TopDocs}. */
    private final int[] sortedDocIds;

    /** Current index in {@link #sortedDocIds}. Starts at -1 before iteration. */
    private int idx = -1;

    /**
     * Constructs a {@code SeededTopDocsDISI} from the given {@link TopDocs}.
     * <p>
     * The document IDs are extracted from {@link org.apache.lucene.search.ScoreDoc#doc}
     * and sorted in ascending order. The collector's base offset, if any, is already removed.
     *
     * @param topDocs the {@link TopDocs} containing the collected document IDs
     */
    public SeededTopDocsDISI(final TopDocs topDocs) {
        sortedDocIds = new int[topDocs.scoreDocs.length];
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            // Remove the doc base as added by the collector
            sortedDocIds[i] = topDocs.scoreDocs[i].doc;
        }
        Arrays.sort(sortedDocIds);
    }

    /**
     * Advances to the first document which is greater than or equals to the current one whose ID is
     * greater than or equal to the given target.
     * <p>
     * This implementation delegates to {@link #slowAdvance(int)} for simplicity.
     *
     * @param target the target document ID
     * @return the next matching document ID, or {@link #NO_MORE_DOCS} if none remain
     * @throws IOException never thrown (declared for interface compatibility)
     */
    @Override
    public int advance(int target) throws IOException {
        return slowAdvance(target);
    }

    /**
     * Returns an estimate of the number of documents this iterator will traverse.
     *
     * @return the number of document IDs available
     */
    @Override
    public long cost() {
        return sortedDocIds.length;
    }

    /**
     * Returns the current document ID.
     *
     * @return the current doc ID, {@code -1} if not yet started,
     *         or {@link #NO_MORE_DOCS} if iteration is complete
     */
    @Override
    public int docID() {
        if (idx == -1) {
            // Not advanced
            return -1;
        } else if (idx >= sortedDocIds.length) {
            // Exhausted doc ids
            return DocIdSetIterator.NO_MORE_DOCS;
        } else {
            return sortedDocIds[idx];
        }
    }

    /**
     * Advances to the next document ID in sorted order.
     *
     * @return the next document ID, or {@link #NO_MORE_DOCS} if there are no more
     */
    @Override
    public int nextDoc() {
        idx += 1;
        return docID();
    }
}
