/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec.halffloatcodec;

import java.io.IOException;
import java.io.UncheckedIOException;

import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;

class KNNHalfFloatByteVectorValues extends ByteVectorValues {
    private final OffHeapByteVectorValues base;
    private final int dim;
    private final int byteSize;
    private final byte[] bytesBuffer;
    private final IndexInput slice;

    KNNHalfFloatByteVectorValues(OffHeapByteVectorValues base, int dim) {
        this.base = base;
        this.dim = dim;
        this.byteSize = dim * Short.BYTES;
        this.bytesBuffer = new byte[byteSize];
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
    public byte[] vectorValue(int ord) throws IOException {
        slice.seek((long) ord * byteSize);
        slice.readBytes(bytesBuffer, 0, byteSize);
        return bytesBuffer;
    }

    @Override
    public ByteVectorValues copy() {
        try {
            return new KNNHalfFloatByteVectorValues((OffHeapByteVectorValues) base.copy(), dim);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public VectorScorer scorer(byte[] query) throws IOException {
        return base.scorer(query);
    }
}
