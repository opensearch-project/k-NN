/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfHalfFloatsSerializer;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Merge-time view that hands {@link KNN1040HalfFloatFlatVectorsWriter} each vector's stored FP16
 * bytes instead of a decoded {@code float[]}, removing one decode and one re-encode per vector.
 *
 * Mirrors {@code MergedVectorValues#mergeFloatVectorValues}: the same {@link DocIDMerger} over the
 * same per-segment subs, so doc order, dropped deletions and index sorting match Lucene's. It is
 * reimplemented rather than wrapped because Lucene's merged view exposes only {@code float[]} and
 * keeps its current sub package-private. A plugin cannot read package-private Lucene members at all -
 * the plugin classloader puts it in a different runtime package, so such access compiles but throws
 * {@link IllegalAccessError} on the first merge, killing the node.
 *
 * The round trip it replaces is exact - FP16 widening is lossless and narrowing recovers every
 * finite value's bit pattern; NaN is rejected at index time. Segments that don't unwrap to FP16 fall
 * back to decode-then-encode: unreachable today, and it costs only that segment the fast path.
 */
class MergeOptimizedHalfFloatVector extends FloatVectorValues {

    private final List<Sub> subs;
    private final DocIDMerger<Sub> docIdMerger;
    private final int size;
    private final int dimension;
    private int docId = -1;
    private int lastOrd = -1;
    private Sub current;

    /**
     * Collects one {@link Sub} per segment holding this field, in the order
     * {@code KnnVectorsWriter.MergedVectorValues} would.
     */
    static MergeOptimizedHalfFloatVector create(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        final int dimension = fieldInfo.getVectorDimension();
        final List<Sub> subs = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            if (KnnVectorsWriter.MergedVectorValues.hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name) == false) {
                continue;
            }
            final KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
            if (reader == null) {
                continue;
            }
            final FloatVectorValues values = reader.getFloatVectorValues(fieldInfo.name);
            if (values == null) {
                continue;
            }
            subs.add(new Sub(mergeState.docMaps[i], values, dimension));
        }
        return new MergeOptimizedHalfFloatVector(subs, mergeState, dimension);
    }

    private MergeOptimizedHalfFloatVector(List<Sub> subs, MergeState mergeState, int dimension) throws IOException {
        this.subs = subs;
        this.dimension = dimension;
        this.docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
        int totalSize = 0;
        for (Sub sub : subs) {
            totalSize += sub.values.size();
        }
        this.size = totalSize;
    }

    /**
     * Fills {@code dest} with the FP16 encoding of the vector at the iterator's current position -
     * copied verbatim when the source segment already stores FP16, encoded from {@code float[]}
     * otherwise. Valid only after {@link #iterator()}'s {@code nextDoc} has returned a real doc.
     *
     * @param dest buffer of exactly {@code dimension * Short.BYTES} bytes
     */
    void currentVectorBytes(byte[] dest) throws IOException {
        if (current == null) {
            throw new IllegalStateException("nextDoc must be called before reading a vector");
        }
        if (current.halfFloatValues != null) {
            current.halfFloatValues.readRawVectorBytes(current.iterator.index(), dest, 0);
            return;
        }
        final float[] vector = current.values.vectorValue(current.iterator.index());
        KNNVectorAsCollectionOfHalfFloatsSerializer.INSTANCE.floatToByteArray(vector, dest, dimension);
    }

    @Override
    public DocIndexIterator iterator() {
        return new DocIndexIterator() {
            private int index = -1;

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int index() {
                return index;
            }

            @Override
            public int nextDoc() throws IOException {
                current = docIdMerger.next();
                if (current == null) {
                    docId = NO_MORE_DOCS;
                    index = NO_MORE_DOCS;
                } else {
                    docId = current.mappedDocID;
                    ++lastOrd;
                    ++index;
                }
                return docId;
            }

            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return size;
            }
        };
    }

    /**
     * Kept so this remains a usable {@link FloatVectorValues}; the merge itself goes through
     * {@link #currentVectorBytes} instead and never decodes.
     */
    @Override
    public float[] vectorValue(int ord) throws IOException {
        if (ord != lastOrd) {
            throw new IllegalStateException("only supports forward iteration with a single iterator: ord=" + ord + ", lastOrd=" + lastOrd);
        }
        return current.values.vectorValue(current.iterator.index());
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int ordToDoc(int ord) {
        throw new UnsupportedOperationException();
    }

    @Override
    public VectorScorer scorer(float[] target) {
        throw new UnsupportedOperationException();
    }

    @Override
    public FloatVectorValues copy() {
        throw new UnsupportedOperationException();
    }

    /**
     * One segment being merged, holding the FP16 values to copy from when this segment was written
     * by {@link KNN1040HalfFloatFlatVectorsWriter}.
     */
    private static final class Sub extends DocIDMerger.Sub {
        final FloatVectorValues values;
        final KnnVectorValues.DocIndexIterator iterator;
        // Null when this segment isn't FP16-backed, which sends currentVectorBytes down its
        // decode-then-encode fallback for this sub only.
        final KNN1040HalfFloatFlatVectorsValues halfFloatValues;

        Sub(MergeState.DocMap docMap, FloatVectorValues values, int dimension) {
            super(docMap);
            this.values = values;
            this.iterator = values.iterator();
            this.halfFloatValues = unwrapHalfFloatValues(values, dimension);
        }

        @Override
        public int nextDoc() throws IOException {
            return iterator.nextDoc();
        }
    }

    /**
     * Peels off the wrappers a reader may return - {@link WrappedFloatVectorValues} layers and the
     * {@link MMapFloatVectorValues} added for mmap-backed segments - to reach the FP16 values
     * underneath, or null if this segment isn't FP16-backed.
     *
     * The dimension check guards the raw copy's buffer: a mismatch would have {@code readRawVectorBytes}
     * write more bytes than the caller's buffer holds. Field metadata should already rule that out, so a
     * mismatch takes the safe fallback rather than throwing.
     */
    private static KNN1040HalfFloatFlatVectorsValues unwrapHalfFloatValues(FloatVectorValues values, int dimension) {
        FloatVectorValues candidate = values;
        while (candidate != null) {
            if (candidate instanceof KNN1040HalfFloatFlatVectorsValues halfFloatValues) {
                return halfFloatValues.dimension() == dimension ? halfFloatValues : null;
            }
            if (candidate instanceof MMapFloatVectorValues mmapValues) {
                candidate = mmapValues.getDelegate();
                continue;
            }
            final FloatVectorValues unwrapped = WrappedFloatVectorValues.getBottomFloatVectorValues(candidate);
            if (unwrapped == candidate) {
                return null;
            }
            candidate = unwrapped;
        }
        return null;
    }
}
