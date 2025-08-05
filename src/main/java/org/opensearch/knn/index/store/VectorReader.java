/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;

public class VectorReader {
    private final KnnVectorValues knnVectorValues;
    private final KnnVectorValues.DocIndexIterator iterator;

    public VectorReader(KnnVectorValues knnVectorValues) {
        this.knnVectorValues = knnVectorValues;
        this.iterator = knnVectorValues.iterator();
    }

    public float[] nextFloatVector() throws IOException {
        int docId = iterator.nextDoc();
        if (docId != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS && knnVectorValues instanceof FloatVectorValues) {
            return ((FloatVectorValues) knnVectorValues).vectorValue(docId);
        }
        return null;
    }

    public byte[] nextByteVector() throws IOException {
        int docId = iterator.nextDoc();
        if (docId != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS && knnVectorValues instanceof ByteVectorValues) {
            return ((ByteVectorValues) knnVectorValues).vectorValue(docId);
        }
        return null;
    }
}
