/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.NonNull;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * An abstract class that provides an iterator to iterate over KNNVectors, as KNNVectors are stored as different
 * representation like {@link BinaryDocValues}, {@link FloatVectorValues}, FieldWriter etc. How to iterate using this
 * iterator please refer {@link DocIdsIteratorValues} java docs.
 */
public interface KNNVectorValuesIterator {

    /**
     * Returns the current docId where the iterator is pointing to.
     * @return int
     */
    int docId();

    /**
     * Advances to a specific docId. Ensure that the passed docId is greater than current docId where Iterator is
     * pointing to, otherwise
     * {@link IOException} will be thrown
     * @return int
     * @throws IOException if we are not able to move to the passed docId.
     */
    int advance(int docId) throws IOException;

    /**
     * Move to nextDocId. If no more docs are present then {@link DocIdSetIterator#NO_MORE_DOCS} will be returned.
     * @return int
     * @throws IOException if we cannot move to next docId
     */
    int nextDoc() throws IOException;

    /**
     * Return a {@link DocIdSetIterator}
     * @return {@link DocIdSetIterator}
     */
    DocIdSetIterator getDocIdSetIterator();

    /**
     * Total number of live doc which will the iterator will iterate upon.
     * @return long: total number of live docs
     */
    long liveDocs();

    /**
     * Returns the {@link VectorValueExtractorStrategy} to extract the vector from the iterator.
     * @return VectorValueExtractorStrategy
     */
    VectorValueExtractorStrategy getVectorExtractorStrategy();

    /**
     * A DocIdsIteratorValues provides a common iteration logic for all Values that implements
     * {@link DocIdSetIterator} interface. Example: {@link BinaryDocValues}, {@link FloatVectorValues} etc.
     */
    class DocIdsIteratorValues implements KNNVectorValuesIterator {
        protected DocIdSetIterator docIdSetIterator;
        private static final List<Function<DocIdSetIterator, Boolean>> VALID_ITERATOR_INSTANCE = List.of(
            (itr) -> itr instanceof BinaryDocValues,
            (itr) -> itr instanceof FloatVectorValues,
            (itr) -> itr instanceof ByteVectorValues
        );

        DocIdsIteratorValues(@NonNull final DocIdSetIterator docIdSetIterator) {
            validateIteratorType(docIdSetIterator);
            this.docIdSetIterator = docIdSetIterator;
        }

        @Override
        public int docId() {
            return docIdSetIterator.docID();
        }

        @Override
        public int advance(int docId) throws IOException {
            return docIdSetIterator.advance(docId);
        }

        @Override
        public int nextDoc() throws IOException {
            return docIdSetIterator.nextDoc();
        }

        @Override
        public DocIdSetIterator getDocIdSetIterator() {
            return docIdSetIterator;
        }

        @Override
        public long liveDocs() {
            if (docIdSetIterator instanceof BinaryDocValues) {
                return KNNCodecUtil.getTotalLiveDocsCount((BinaryDocValues) docIdSetIterator);
            } else if (docIdSetIterator instanceof FloatVectorValues || docIdSetIterator instanceof ByteVectorValues) {
                return docIdSetIterator.cost();
            }
            throw new IllegalArgumentException(
                "DocIdSetIterator present is not of valid type. Valid types are: BinaryDocValues, FloatVectorValues and ByteVectorValues"
            );
        }

        @Override
        public VectorValueExtractorStrategy getVectorExtractorStrategy() {
            return new VectorValueExtractorStrategy.DISIVectorExtractor();
        }

        private void validateIteratorType(final DocIdSetIterator docIdSetIterator) {
            VALID_ITERATOR_INSTANCE.stream()
                .map(v -> v.apply(docIdSetIterator))
                .filter(Boolean::booleanValue)
                .findFirst()
                .orElseThrow(
                    () -> new IllegalArgumentException(
                        "DocIdSetIterator present is not of valid type. Valid types are: BinaryDocValues, FloatVectorValues and ByteVectorValues"
                    )
                );
        }
    }

    /**
     * A FieldWriterIteratorValues is mainly used when Vectors are stored in {@link KnnFieldVectorsWriter} interface.
     */
    class FieldWriterIteratorValues<T> implements KNNVectorValuesIterator {
        private final DocIdSetIterator docIdSetIterator;
        private final Map<Integer, T> vectors;

        FieldWriterIteratorValues(@NonNull final DocsWithFieldSet docsWithFieldSet, @NonNull final Map<Integer, T> vectors) {
            assert docsWithFieldSet.iterator().cost() == vectors.size();
            this.vectors = vectors;
            this.docIdSetIterator = docsWithFieldSet.iterator();
        }

        @Override
        public int docId() {
            return docIdSetIterator.docID();
        }

        @Override
        public int advance(int docId) throws IOException {
            return docIdSetIterator.advance(docId);
        }

        @Override
        public int nextDoc() throws IOException {
            return docIdSetIterator.nextDoc();
        }

        /**
         * Returns a Map of docId and vector.
         * @return {@link Map}
         */
        public T vectorsValue() {
            return vectors.get(docId());
        }

        @Override
        public DocIdSetIterator getDocIdSetIterator() {
            return docIdSetIterator;
        }

        @Override
        public long liveDocs() {
            return docIdSetIterator.cost();
        }

        @Override
        public VectorValueExtractorStrategy getVectorExtractorStrategy() {
            return new VectorValueExtractorStrategy.FieldWriterIteratorVectorExtractor();
        }
    }

    abstract class MergeSegmentVectorValuesIterator<T extends DocIdSetIterator, U> implements KNNVectorValuesIterator {

        private DocIDMerger<KNNMergeVectorValues.KNNVectorValuesSub<T>> docIdMerger;
        private final int liveDocs;
        private int docId;
        protected KNNMergeVectorValues.KNNVectorValuesSub<T> current;

        private static final VectorValueExtractorStrategy VECTOR_VALUES_STRATEGY =
            new VectorValueExtractorStrategy.MergeSegmentValuesExtractor();

        MergeSegmentVectorValuesIterator(final List<KNNMergeVectorValues.KNNVectorValuesSub<T>> subs, final MergeState mergeState)
            throws IOException {
            this.docIdMerger = DocIDMerger.of(subs, mergeState.needsIndexSort);
            int totalSize = 0;
            for (KNNMergeVectorValues.KNNVectorValuesSub<T> sub : subs) {
                totalSize += sub.liveDocs;
            }
            this.liveDocs = totalSize;
            this.docId = -1;

        }

        @Override
        public int docId() {
            return docId;
        }

        @Override
        public int advance(int docId) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public int nextDoc() throws IOException {
            current = docIdMerger.next();
            if (current == null) {
                docId = NO_MORE_DOCS;
            } else {
                docId = current.mappedDocID;
            }
            return docId;
        }

        @Override
        public DocIdSetIterator getDocIdSetIterator() {
            // while we can get the values of current, this method is intended to be called once so it's much better to throw
            // so Liskov-Substitution-principle is not violated unknowingly
            throw new UnsupportedOperationException();
        }

        @Override
        public long liveDocs() {
            return liveDocs;
        }

        @Override
        public VectorValueExtractorStrategy getVectorExtractorStrategy() {
            return VECTOR_VALUES_STRATEGY;
        }

        public abstract U vectorValue() throws IOException;
    }

    class MergeFloat32VectorValuesIterator extends MergeSegmentVectorValuesIterator<FloatVectorValues, float[]> {

        MergeFloat32VectorValuesIterator(
            final List<KNNMergeVectorValues.KNNVectorValuesSub<FloatVectorValues>> subs,
            final MergeState mergeState
        ) throws IOException {
            super(subs, mergeState);
        }

        @Override
        public float[] vectorValue() throws IOException {
            return current.values.vectorValue();
        }
    }

    class MergeByteVectorValuesIterator extends MergeSegmentVectorValuesIterator<ByteVectorValues, byte[]> {

        MergeByteVectorValuesIterator(
            final List<KNNMergeVectorValues.KNNVectorValuesSub<ByteVectorValues>> subs,
            final MergeState mergeState
        ) throws IOException {
            super(subs, mergeState);
        }

        @Override
        public byte[] vectorValue() throws IOException {
            return current.values.vectorValue();
        }
    }
}
