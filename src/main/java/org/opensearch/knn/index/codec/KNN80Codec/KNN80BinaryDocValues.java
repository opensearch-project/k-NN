/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.Getter;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;

/**
 * A per-document kNN numeric value.
 */
public class KNN80BinaryDocValues extends BinaryDocValues {

    private DocIDMerger<BinaryDocValuesSub> docIDMerger;

    @Getter
    private long totalLiveDocs;

    KNN80BinaryDocValues(DocIDMerger<BinaryDocValuesSub> docIdMerger) {
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

    /**
     * Builder pattern like setter for setting totalLiveDocs. We can use setter also. But this way the code is clean.
     * @param totalLiveDocs int
     * @return {@link KNN80BinaryDocValues}
     */
    public KNN80BinaryDocValues setTotalLiveDocs(long totalLiveDocs) {
        this.totalLiveDocs = totalLiveDocs;
        return this;
    }
}
