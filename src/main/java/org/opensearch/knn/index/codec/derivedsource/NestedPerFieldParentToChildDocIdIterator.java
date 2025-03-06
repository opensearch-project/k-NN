/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Getter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Iterator over the children documents of a particular parent
 */
public class NestedPerFieldParentToChildDocIdIterator {

    private final int parentDocId;
    @Getter
    private int firstChild;
    private int currentChild;
    private final KNNVectorValues<?> vectorValues;

    /**
     * Constructor
     *
     * @param parentDocId the parent docId
     * @param firstChild the first child of this parent doc
     * @param vectorValues the vector values
     */
    public NestedPerFieldParentToChildDocIdIterator(int parentDocId, int firstChild, KNNVectorValues<?> vectorValues) {
        this.parentDocId = parentDocId;
        this.vectorValues = vectorValues;
        this.currentChild = -1;
        this.firstChild = firstChild;
    }

    /**
     * Get the next child for this parent
     *
     * @return the next child docId. If there are no more children, return
     * NO_MORE_DOCS
     */
    public int nextDoc() throws IOException {
        if (currentChild == NO_MORE_DOCS) {
            return NO_MORE_DOCS;
        }

        // On the first call, we advance to the first child and, if it has the vector for the field, return it.
        if (currentChild == -1) {
            currentChild = vectorValues.advance(firstChild);
        } else {
            currentChild = vectorValues.nextDoc();
        }

        if (currentChild >= parentDocId) {
            currentChild = NO_MORE_DOCS;
            if (vectorValues.docId() != NO_MORE_DOCS) {
                vectorValues.advance(NO_MORE_DOCS);
            }
        }

        return currentChild;
    }

    /**
     * Get the current child for this parent
     *
     * @return the current child docId. If this has not been set, return -1
     */
    public int docId() {
        return currentChild;
    }

    /**
     * Get the vector value for the current child.
     *
     * @return the vector for the current child
     * @throws IOException if there is an error reading the vector values
     */
    public Object getVector() throws IOException {
        return vectorValues.getVector();
    }

    /**
     * Get a clone of the vector for the current child.
     *
     * @return a clone of the vector for the current child
     * @throws IOException if there is an error reading the vector values
     */
    public Object getVectorClone() throws IOException {
        return vectorValues.conditionalCloneVector();
    }
}
