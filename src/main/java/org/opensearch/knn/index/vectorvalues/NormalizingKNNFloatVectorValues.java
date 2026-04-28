/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.util.Arrays;

/**
 * Wraps a {@link KNNFloatVectorValues} and returns L2-normalized copies of the vectors.
 *
 * <p>This is used by the Faiss native index build path when the space type is cosine similarity.
 * Faiss does not natively support cosine similarity; instead we convert cosine to inner product
 * and feed unit vectors to the index. Storing the original (unnormalized) vectors in doc values
 * while only normalizing at build time preserves the original data for downstream consumers such
 * as derived source reconstruction.
 */
public class NormalizingKNNFloatVectorValues extends KNNFloatVectorValues {

    private final KNNFloatVectorValues delegate;

    public NormalizingKNNFloatVectorValues(final KNNFloatVectorValues delegate) {
        super(delegate.getVectorValuesIterator());
        this.delegate = delegate;
    }

    @Override
    public float[] getVector() throws IOException {
        final float[] original = delegate.getVector();
        // Keep local caches consistent with the delegate (dimension/bytesPerVector are populated on first getVector()).
        this.dimension = delegate.dimension;
        this.bytesPerVector = delegate.bytesPerVector;
        final float[] copy = Arrays.copyOf(original, original.length);
        VectorUtil.l2normalize(copy);
        return copy;
    }

    @Override
    public float[] conditionalCloneVector() throws IOException {
        // getVector() already returns a fresh copy, so no further clone is needed.
        return getVector();
    }
}
