/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN91Codec.docformat;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;

import java.io.IOException;

/**
 * A per-document kNN numeric value.
 */
class KNN91BinaryDocValues extends BinaryDocValues {

    private DocIDMerger<BinaryDocValuesSub> docIDMerger;

    KNN91BinaryDocValues(DocIDMerger<BinaryDocValuesSub> docIdMerger) {
        this.docIDMerger = docIdMerger;
    }

    private BinaryDocValuesSub current;
    private int docID = -1;

    @Override
    public int docID() {
        return docID;
    }

    @Override
    public int nextDoc() throws IOException {
        current = docIDMerger.next();
        if (current == null) {
            docID = NO_MORE_DOCS;
        } else {
            docID = current.mappedDocID;
        }
        return docID;
    }

    @Override
    public int advance(int target) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean advanceExact(int target) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public long cost() {
        throw new UnsupportedOperationException();
    }

    @Override
    public BytesRef binaryValue() throws IOException {
        return current.getValues().binaryValue();
    }
};
