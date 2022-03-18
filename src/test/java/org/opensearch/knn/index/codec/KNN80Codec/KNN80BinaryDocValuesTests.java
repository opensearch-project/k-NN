/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableList;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.MergeState;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.util.BinaryDocValuesSub;

import java.io.IOException;

public class KNN80BinaryDocValuesTests extends KNNTestCase {

    public void testDocId() {
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(null);
        assertEquals(-1, knn80BinaryDocValues.docID());
    }

    public void testNextDoc() throws IOException {
        final int expectedDoc = 12;

        BinaryDocValuesSub sub = new BinaryDocValuesSub(new MergeState.DocMap() {
            @Override
            public int get(int docID) {
                return expectedDoc;
            }
        }, new KNNCodecTestUtil.ConstantVectorBinaryDocValues(10, 128, 1.0f));

        DocIDMerger<BinaryDocValuesSub> docIDMerger = DocIDMerger.of(ImmutableList.of(sub), false);
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(docIDMerger);
        assertEquals(expectedDoc, knn80BinaryDocValues.nextDoc());
    }

    public void testAdvance() {
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(null);
        expectThrows(UnsupportedOperationException.class, () -> knn80BinaryDocValues.advance(0));
    }

    public void testAdvanceExact() {
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(null);
        expectThrows(UnsupportedOperationException.class, () -> knn80BinaryDocValues.advanceExact(0));
    }

    public void testCost() {
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(null);
        expectThrows(UnsupportedOperationException.class, knn80BinaryDocValues::cost);
    }

    public void testBinaryValue() throws IOException {
        BinaryDocValues binaryDocValues = new KNNCodecTestUtil.ConstantVectorBinaryDocValues(10, 128, 1.0f);
        BinaryDocValuesSub sub = new BinaryDocValuesSub(new MergeState.DocMap() {
            @Override
            public int get(int docID) {
                return docID;
            }
        }, binaryDocValues);

        DocIDMerger<BinaryDocValuesSub> docIDMerger = DocIDMerger.of(ImmutableList.of(sub), false);
        KNN80BinaryDocValues knn80BinaryDocValues = new KNN80BinaryDocValues(docIDMerger);
        knn80BinaryDocValues.nextDoc();
        assertEquals(binaryDocValues.binaryValue(), knn80BinaryDocValues.binaryValue());
    }
}
