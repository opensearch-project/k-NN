/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

/**
 * A {@link DocIdSetIterator} that maps document IDs from a source iterator
 * to their corresponding vector indices in a {@link KnnVectorValues.DocIndexIterator}.
 * <p>
 * This class advances the {@code indexedDISI} (which provides access to vector indices)
 * in sync with the {@code sourceDISI} (which provides document IDs). For each document ID
 * emitted by the source iterator, it advances the index iterator to the same document ID
 * and returns the associated vector index.
 * <p>
 * Typical usage is when document-level matches (from a collector or filter)
 * need to be mapped back to the vector index space for further vector-based operations.
 */
public class SeededMappedDISI extends DocIdSetIterator {

    // Iterator over vector values that exposes both doc IDs and their corresponding indices.
    private final KnnVectorValues.DocIndexIterator indexedDISI;

    // Source iterator that provides the sequence of document IDs to be mapped.
    private final DocIdSetIterator sourceDISI;

    /**
     * Constructs a {@code SeededMappedDISI} that synchronizes a source document iterator
     * with a vector index iterator.
     *
     * @param indexedDISI the {@link KnnVectorValues.DocIndexIterator} used to map
     *                    document IDs to vector indices
     * @param sourceDISI the {@link DocIdSetIterator} providing the source document IDs
     */
    public SeededMappedDISI(KnnVectorValues.DocIndexIterator indexedDISI, DocIdSetIterator sourceDISI) {
        this.indexedDISI = indexedDISI;
        this.sourceDISI = sourceDISI;
    }

    /**
     * Advances the source iterator to the first document ID that is greater than or equal
     * to the specified target, then advances the index iterator to the same document ID.
     * <p>
     * The returned value is the vector index corresponding to that document.
     *
     * @param target the target document ID
     * @return the corresponding vector index, or {@link #NO_MORE_DOCS} if the end is reached
     * @throws IOException if an I/O error occurs
     */
    @Override
    public int advance(int target) throws IOException {
        int newTarget = sourceDISI.advance(target);
        if (newTarget != NO_MORE_DOCS) {
            indexedDISI.advance(newTarget);
        }
        return docID();
    }

    /**
     * Returns an estimate of the cost (number of documents) of iterating.
     *
     * @return the cost estimate from the source iterator
     */
    @Override
    public long cost() {
        return sourceDISI.cost();
    }

    /**
     * Returns the current vector index corresponding to the current document position.
     *
     * @return the current vector index, or {@link #NO_MORE_DOCS} if iteration has completed
     */
    @Override
    public int docID() {
        if (indexedDISI.docID() == NO_MORE_DOCS || sourceDISI.docID() == NO_MORE_DOCS) {
            return NO_MORE_DOCS;
        }
        return indexedDISI.index();
    }

    /**
     * Advances to the next document in the source iterator and updates the index iterator
     * to the same document. Returns the corresponding vector index.
     *
     * @return the next vector index, or {@link #NO_MORE_DOCS} if there are no more documents
     * @throws IOException if an I/O error occurs
     */
    @Override
    public int nextDoc() throws IOException {
        int newTarget = sourceDISI.nextDoc();
        if (newTarget != NO_MORE_DOCS) {
            indexedDISI.advance(newTarget);
        }
        return docID();
    }
}
