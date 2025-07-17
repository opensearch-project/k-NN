/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import java.io.IOException;

public class VectorReader {
    private final FloatVectorValues floatVectorValues;
    private final DocIndexIterator iterator;

    public VectorReader(FloatVectorValues floatVectorValues) {
        this.floatVectorValues = floatVectorValues;
        this.iterator = floatVectorValues.iterator();
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
    public float[] next() throws IOException {
        int docId = iterator.nextDoc();
        if (docId != DocIndexIterator.NO_MORE_DOCS) {
            return floatVectorValues.vectorValue(docId);
        } else {
            return null;
        }
    }
}