/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class VectorReader {
    private final KNNVectorValues<?> knnVectorValues;

    public VectorReader(KNNVectorValues<?> knnVectorValues) {
        this.knnVectorValues = knnVectorValues;
    }

    /**
     * Fetches the next float vector from the iterator.
     *
     * This method will be invoked by the native layer via JNI to sequentially
     * retrieve vectors during index loading.
     *
     * @return The next available float vector, or null
     * @throws IOException if there is an error accessing the vector
     */
    public float[] nextFloatVector() throws IOException {
        int docId = knnVectorValues.nextDoc();
        if (docId != NO_MORE_DOCS && knnVectorValues instanceof KNNFloatVectorValues) {
            return ((KNNFloatVectorValues) knnVectorValues).getVector();
        }
        return null;
    }

    /**
     * Fetches the next byte vector from the iterator.
     *
     * This method will be invoked by the native layer via JNI to sequentially
     * retrieve vectors during index loading.
     *
     * @return The next available byte vector, or null
     * @throws IOException if there is an error accessing the vector
     */
    public byte[] nextByteVector() throws IOException {
        int docId = knnVectorValues.nextDoc();
        if (docId != NO_MORE_DOCS && knnVectorValues instanceof KNNByteVectorValues) {
            return ((KNNByteVectorValues) knnVectorValues).getVector();
        }
        return null;
    }
}
