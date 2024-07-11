/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.AllArgsConstructor;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;
import java.util.Iterator;

/**
 * An abstract class that prvides an iterator to iterate over KNNVectors, as KNNVectors are stored as different
 * representation like BinaryDocValues, FloatVectorValues, FieldWriter etc.
 */
public interface KNNVectorValuesIterator {

    int docId();

    int advance(int docId) throws IOException;

    int nextDoc() throws IOException;

    DocIdSetIterator getDocIdSetIterator();

    @AllArgsConstructor
    class DocIdsIteratorValues implements KNNVectorValuesIterator {
        protected DocIdSetIterator docIdSetIterator;

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
    }

    @AllArgsConstructor
    class FieldWriterIteratorValues<T> implements KNNVectorValuesIterator {
        protected DocIdSetIterator docIdSetIterator;
        protected Iterator<T> vectorsIterator;

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

        public Iterator<T> getVectorValues() {
            return vectorsIterator;
        }

        @Override
        public DocIdSetIterator getDocIdSetIterator() {
            return docIdSetIterator;
        }
    }

}
