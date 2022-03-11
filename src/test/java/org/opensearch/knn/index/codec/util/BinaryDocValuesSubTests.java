/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.util;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.MergeState;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

import java.io.IOException;

public class BinaryDocValuesSubTests extends KNNTestCase {

    public void testNextDoc() throws IOException {
        BinaryDocValues binaryDocValues = new KNNCodecTestUtil.ConstantVectorBinaryDocValues(10, 128, 2.0f);
        MergeState.DocMap docMap = new MergeState.DocMap() {
            @Override
            public int get(int docID) {
                return docID;
            }
        };

        BinaryDocValuesSub binaryDocValuesSub = new BinaryDocValuesSub(docMap, binaryDocValues);
        int expectedNextDoc = binaryDocValues.nextDoc() + 1;
        assertEquals(expectedNextDoc, binaryDocValuesSub.nextDoc());
    }

    public void testGetValues() {
        BinaryDocValues binaryDocValues = new KNNCodecTestUtil.ConstantVectorBinaryDocValues(10, 128, 2.0f);
        MergeState.DocMap docMap = new MergeState.DocMap() {
            @Override
            public int get(int docID) {
                return docID;
            }
        };

        BinaryDocValuesSub binaryDocValuesSub = new BinaryDocValuesSub(docMap, binaryDocValues);

        assertEquals(binaryDocValues, binaryDocValuesSub.getValues());
    }

}
