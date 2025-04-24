/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import com.google.common.annotations.VisibleForTesting;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Utility class used to implement Lucene functionality that can be used to interact with Lucene
 */
@RequiredArgsConstructor
public class DerivedSourceLuceneHelper {

    // Offsets from the parent docId to start from in order to determine where to start the search from for first child.
    // In other words, we're guessing the upperbound on the number of nested documents a single parent will have.
    // This is an optimization to avoid starting from the first doc to find the previous parent to the parent docid.
    // The values are just back of the napkin calculations, but heres how I got these numbers: Assuming there are
    // ~12 bytes an entry for NumericDocValues (8 for long value, 4 for int id). On a single
    // 4kb page, it should be possible to fit 340 values. If first offset is 150, we can be confident it will be only 1
    // page fetched typically. Then 10 pages and then 40 pages.
    private static final int[] NESTED_OFFSET_STARTING_POINTS = new int[] { 150, 1500, 6000 };

    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;

    @VisibleForTesting
    static final int NO_CHILDREN_INDICATOR = -1;

    /**
     * Get the first child of the given parentDoc. This can be used to determine if the document contains any nested
     * fields.
     *
     * @param parentDocId Parent doc id to find children for
     * @return doc id of last matching doc. {@link DocIdSetIterator#NO_MORE_DOCS} if no children exist.
     * @throws IOException
     */
    public int getFirstChild(int parentDocId) throws IOException {
        // If its the first document id, then there is no change there are parents
        if (parentDocId == 0) {
            return NO_MORE_DOCS;
        }
        int lastStartingPoint = -1;
        for (int offset : NESTED_OFFSET_STARTING_POINTS) {
            int currentStartingPoint = Math.max(0, parentDocId - offset);
            // If we've already checked this starting point, no need to continue
            if (currentStartingPoint <= lastStartingPoint) {
                break;
            }
            int firstChild = getFirstChild(parentDocId, currentStartingPoint);
            // If the returned value is NO_CHILDREN_INDICATOR, we know for sure that there are no children. No need to
            // keep checking
            if (firstChild == NO_CHILDREN_INDICATOR) {
                return NO_MORE_DOCS;
            }
            // If the first child is in between currentStartingPoint and parentDocId, we can return
            if (firstChild != NO_MORE_DOCS) {
                return firstChild;
            }
            lastStartingPoint = currentStartingPoint;
        }
        // If none of the shortcuts worked, we'll try from the start
        return getFirstChild(parentDocId, 0);
    }

    @VisibleForTesting
    int getFirstChild(int parentDocId, int startingPoint) throws IOException {
        // Only root level documents have the "_primary_term" field. So, we iterate through all of the documents in
        // order to find out if any have this term.
        FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo("_primary_term");
        assert derivedSourceReaders.getDocValuesProducer() != null;
        NumericDocValues numericDocValues = derivedSourceReaders.getDocValuesProducer().getNumeric(fieldInfo);
        int previousParentDocId = NO_MORE_DOCS;
        numericDocValues.advance(startingPoint);
        while (numericDocValues.docID() != NO_MORE_DOCS) {
            if (numericDocValues.docID() >= parentDocId) {
                break;
            }
            previousParentDocId = numericDocValues.docID();
            numericDocValues.nextDoc();
        }

        // If there are no numeric docvalues before the current parent doc, then the parent doc is the first parent. So
        // its first child must be 0
        if (previousParentDocId == NO_MORE_DOCS) {
            return 0;
        }
        // If the document right before is the previous parent, then there are no children. Return
        if (parentDocId - previousParentDocId <= 1) {
            return NO_CHILDREN_INDICATOR;
        }
        return previousParentDocId + 1;
    }
}
