/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec.halffloatcodec;

import java.io.IOException;
import java.io.UncheckedIOException;

import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;

class KNNHalfFloatVectorValues extends FloatVectorValues {
    private final OffHeapFloatVectorValues base;
    private final int dim;
    private final int byteSize;
    private final byte[] bytesBuffer;
    private final float[] floatBuffer;
    private final IndexInput slice;

    KNNHalfFloatVectorValues(OffHeapFloatVectorValues base, int dim) {
        this.base = base;
        this.dim = dim;
        this.byteSize = dim * Short.BYTES;
        this.bytesBuffer = new byte[byteSize];
        this.floatBuffer = new float[dim];
        this.slice = base.getSlice();
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public int size() {
        return base.size();
    }

    @Override
    public int ordToDoc(int ord) {
        return base.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits bits) {
        return base.getAcceptOrds(bits);
    }

    @Override
    public DocIndexIterator iterator() {
        return base.iterator();
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        slice.seek((long) ord * byteSize);
        slice.readBytes(bytesBuffer, 0, byteSize);
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.byteToFloatArray(bytesBuffer, floatBuffer, dim, 0);
        return floatBuffer;
    }

    @Override
    public FloatVectorValues copy() {
        try {
            return new KNNHalfFloatVectorValues((OffHeapFloatVectorValues) base.copy(), dim);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
        return base.scorer(query);
    }
}
