/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import java.io.IOException;

public class VectorReader {
    private final KnnVectorValues knnVectorValues;
    private final DocIndexIterator iterator;

    public VectorReader(KnnVectorValues knnVectorValues) {
        this.knnVectorValues = knnVectorValues;
        this.iterator = knnVectorValues.iterator();
    }

    /**
     * Fetches the next float vector from the iterator.
     *
     * This method will be invoked by the native layer via JNI to sequentially
     * retrieve vectors during index loading or search-time deserialization.
     *
     * @return The next available float vector, or null if no more vectors are available.
     * @throws IOException if reading the vector value fails.
     */
    public float[] nextFloatVector() throws IOException {
        int docId = iterator.nextDoc();
        if (docId != DocIndexIterator.NO_MORE_DOCS && knnVectorValues instanceof FloatVectorValues) {
            return ((FloatVectorValues) knnVectorValues).vectorValue(docId);
        }
        return null;
    }

    /**
     * Fetches the next byte vector from the iterator.
     *
     * This method will be invoked by the native layer via JNI to sequentially
     * retrieve byte vectors during index loading or search-time deserialization.
     *
     * @return The next available byte vector, or null if no more vectors are available.
     * @throws IOException if reading the vector value fails.
     */
    public byte[] nextByteVector() throws IOException {
        int docId = iterator.nextDoc();
        if (docId != DocIndexIterator.NO_MORE_DOCS && knnVectorValues instanceof ByteVectorValues) {
            return ((ByteVectorValues) knnVectorValues).vectorValue(docId);
        }
        return null;
    }

}
