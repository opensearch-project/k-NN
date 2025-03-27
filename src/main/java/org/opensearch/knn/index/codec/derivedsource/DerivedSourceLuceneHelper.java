/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

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

    private final DerivedSourceReaders derivedSourceReaders;
    private final SegmentReadState segmentReadState;

    /**
     * Get the first child of the given parentDoc. This can be used to determine if the document contains any nested
     * fields.
     *
     * @return doc id of last matching doc. {@link DocIdSetIterator#NO_MORE_DOCS} if no children exist.
     * @throws IOException
     */
    public int getFirstChild(int parentDocId) throws IOException {
        // If its the first document id, then there is no change there are parents
        if (parentDocId == 0) {
            return NO_MORE_DOCS;
        }

        // Only root level documents have the "_primary_term" field. So, we iterate through all of the documents in
        // order to find out if any have this term.
        // TODO: This is expensive and should be optimized. We should start at doc parentDocId - 10000 and work back
        // (can we fetch the setting? Maybe)
        FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo("_primary_term");
        assert derivedSourceReaders.getDocValuesProducer() != null;
        NumericDocValues numericDocValues = derivedSourceReaders.getDocValuesProducer().getNumeric(fieldInfo);
        int previousParentDocId = NO_MORE_DOCS;
        numericDocValues.advance(0);
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
        // If the document right before is the previous parent, then there are no children.
        if (parentDocId - previousParentDocId <= 1) {
            return NO_MORE_DOCS;
        }
        return previousParentDocId + 1;
    }
}
