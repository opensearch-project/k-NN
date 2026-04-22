/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.index.KnnVectorValues;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

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
     * Abstract base class for KNN vector iterators, encapsulating common iteration logic.
     */
    abstract class AbstractVectorValuesIterator implements KNNVectorValuesIterator {
        protected final DocIdSetIterator docIdSetIterator;

        AbstractVectorValuesIterator(@NonNull final DocIdSetIterator docIdSetIterator) {
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
            return docIdSetIterator.cost();
        }
    }

    /**
     * A DocIdsIteratorValues provides a common iteration logic for all Values that implements
     * {@link DocIdSetIterator} interface. Example: {@link BinaryDocValues}, {@link FloatVectorValues} etc.
     */
    @Getter
    class DocIdsIteratorValues extends AbstractVectorValuesIterator {

        private final KnnVectorValues knnVectorValues;

        @Setter
        private int lastOrd = -1;
        @Setter
        private Object lastAccessedVector = null;

        DocIdsIteratorValues(@NonNull final KnnVectorValues knnVectorValues) {
            super(knnVectorValues.iterator());
            this.knnVectorValues = knnVectorValues;
        }

        DocIdsIteratorValues(@NonNull final DocIdSetIterator docIdSetIterator) {
            super(docIdSetIterator);
            this.knnVectorValues = null;
        }

        DocIdsIteratorValues(@NonNull final DocIdSetIterator docIdSetIterator, @NonNull final KnnVectorValues knnVectorValues) {
            super(docIdSetIterator);
            this.knnVectorValues = knnVectorValues;
        }

        @Override
        public long liveDocs() {
            if (docIdSetIterator instanceof BinaryDocValues) {
                return KNNCodecUtil.getTotalLiveDocsCount((BinaryDocValues) docIdSetIterator);
            } else if (docIdSetIterator instanceof KnnVectorValues.DocIndexIterator) {
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

    }

    /**
     * A FieldWriterIteratorValues is mainly used when Vectors are stored in {@link KnnFieldVectorsWriter} interface.
     */
    class FieldWriterIteratorValues<T> extends AbstractVectorValuesIterator {
        private final Function<Integer, T> vectorGetter;
        private int index = -1;
        private int lastDocId = Integer.MIN_VALUE;

        FieldWriterIteratorValues(@NonNull final DocsWithFieldSet docsWithFieldSet, @NonNull final Map<Integer, T> vectors) {
            super(docsWithFieldSet.iterator());
            assert docsWithFieldSet.iterator().cost() == vectors.size();
            this.vectorGetter = vectors::get;
        }

        FieldWriterIteratorValues(@NonNull final DocsWithFieldSet docsWithFieldSet, @NonNull final List<T> vectors) {
            super(docsWithFieldSet.iterator());
            assert docsWithFieldSet.iterator().cost() == vectors.size();
            // We can return vectors in sequential manner.
            // Dense case -> Easy. doc_id == vector_ordinal and doc_id will be given as 0, 1, ..., N - 1
            // Sparse case -> doc_id will be given in increasing order 1, 4, 7, 8, 10, ...
            // but its corresponding vector ordinal is 0, 1, 2, ...
            // The getter is idempotent — multiple getVector() calls for the same doc return the same vector
            this.vectorGetter = (docId) -> {
                if (docId != this.lastDocId) {
                    if (docId < this.lastDocId) {
                        throw new IllegalStateException(
                            "Doc IDs must be in increasing order, but got " + docId + " after " + this.lastDocId
                        );
                    }
                    this.index++;
                    this.lastDocId = docId;
                }
                return vectors.get(this.index);
            };
        }

        /**
         * Returns a Map of docId and vector.
         * @return {@link Map}
         */
        public T vectorsValue() {
            return vectorGetter.apply(docId());
        }

        @Override
        public VectorValueExtractorStrategy getVectorExtractorStrategy() {
            return new VectorValueExtractorStrategy.FieldWriterIteratorVectorExtractor();
        }
    }
}
