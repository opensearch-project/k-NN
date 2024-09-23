/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class to get merged VectorValues from MergeState
 */
public final class KNNMergeVectorValues {

    /**
     * Gets list of {@link KNNVectorValuesSub} for {@link FloatVectorValues} from a merge state and returns the iterator which
     * iterates over live docs from all segments while mapping docIds.
     *
     * @param fieldInfo
     * @param mergeState
     * @return List of KNNVectorSub
     * @throws IOException
     */
    public static KNNVectorValuesIterator.MergeFloat32VectorValuesIterator mergeFloatVectorValues(
        FieldInfo fieldInfo,
        MergeState mergeState
    ) throws IOException {
        assert fieldInfo != null && fieldInfo.hasVectorValues();
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new UnsupportedOperationException("Cannot merge vectors encoded as [" + fieldInfo.getVectorEncoding() + "] as FLOAT32");
        }
        final List<KNNVectorValuesSub<FloatVectorValues>> subs = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader != null) {
                FloatVectorValues values = knnVectorsReader.getFloatVectorValues(fieldInfo.getName());
                if (values != null) {
                    final Bits liveDocs = mergeState.liveDocs[i];
                    final int liveDocsCardinality = cardinality(liveDocs, Math.toIntExact(values.cost()));
                    subs.add(new KNNVectorValuesSub<>(mergeState.docMaps[i], values, liveDocsCardinality));
                }
            }
        }
        return new KNNVectorValuesIterator.MergeFloat32VectorValuesIterator(subs, mergeState);
    }

    /**
     * Gets list of {@link KNNVectorValuesSub} for {@link ByteVectorValues} from a merge state. This can be further
     * used to create an iterator for getting the docs and its vector values
     * @param fieldInfo
     * @param mergeState
     * @return List of KNNVectorSub
     * @throws IOException
     */
    public static KNNVectorValuesIterator.MergeByteVectorValuesIterator mergeByteVectorValues(FieldInfo fieldInfo, MergeState mergeState)
        throws IOException {
        assert fieldInfo != null && fieldInfo.hasVectorValues();
        if (fieldInfo.getVectorEncoding() != VectorEncoding.BYTE) {
            throw new UnsupportedOperationException("Cannot merge vectors encoded as [" + fieldInfo.getVectorEncoding() + "] as BYTE");
        }
        final List<KNNVectorValuesSub<ByteVectorValues>> subs = new ArrayList<>();
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader != null) {
                ByteVectorValues values = knnVectorsReader.getByteVectorValues(fieldInfo.getName());
                if (values != null) {
                    final Bits liveDocs = mergeState.liveDocs[i];
                    final int liveDocsCardinality = cardinality(liveDocs, Math.toIntExact(values.cost()));
                    subs.add(new KNNVectorValuesSub<>(mergeState.docMaps[i], values, liveDocsCardinality));
                }
            }
        }
        return new KNNVectorValuesIterator.MergeByteVectorValuesIterator(subs, mergeState);
    }

    private static int cardinality(final Bits liveDocs, final int defaultCount) {
        if (liveDocs == null) {
            return defaultCount;
        }

        if (liveDocs instanceof FixedBitSet) {
            return ((FixedBitSet) liveDocs).cardinality();
        }

        int count = 0;
        for (int index = 0; index < liveDocs.length(); index++) {
            count += liveDocs.get(index) ? 1 : 0;
        }
        return count;
    }

    static class KNNVectorValuesSub<T extends DocIdSetIterator> extends DocIDMerger.Sub {
        final T values;
        final int liveDocs;

        KNNVectorValuesSub(MergeState.DocMap docMap, T values, int liveDocs) {
            super(docMap);
            this.values = values;
            this.liveDocs = liveDocs;
        }

        @Override
        public int nextDoc() throws IOException {
            return values.nextDoc();
        }
    }
}
