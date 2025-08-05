/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

public class KnnFloatVectorValuesAdapter extends FloatVectorValues {

    private final KNNVectorValues<float[]> knnVectorValues;

    public KnnFloatVectorValuesAdapter(KNNVectorValues<float[]> knnVectorValues) {
        this.knnVectorValues = knnVectorValues;
    }

    @Override
    public int size() {
        throw new UnsupportedOperationException("size() not supported; iterate instead");
    }

    @Override
    public int dimension() {
        return knnVectorValues.dimension();
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        return knnVectorValues.getVector();
    }

    @Override
    public FloatVectorValues copy() {
        return new KnnFloatVectorValuesAdapter(knnVectorValues);
    }

    @Override
    public DocIndexIterator iterator() {
        return new DocIndexIterator() {
            @Override
            public int docID() {
                return knnVectorValues.docId();
            }

            @Override
            public int index() {
                return docID();
            }

            @Override
            public int nextDoc() throws IOException {
                return knnVectorValues.nextDoc();
            }

            @Override
            public int advance(int target) throws IOException {
                return knnVectorValues.advance(target);
            }

            @Override
            public long cost() {
                return 0; // could be improved
            }
        };
    }
}
