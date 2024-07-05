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

import lombok.Getter;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * An abstract class to iterate over KNNVectors, as KNNVectors are stored as different representation like
 * BinaryDocValues, FloatVectorValues, FieldWriter etc.
 * @param <T>
 */
public abstract class KNNVectorValues<T> {

    @Getter
    protected KNNVectorValuesIterator vectorValuesIterator;
    protected int dimension;

    protected KNNVectorValues(KNNVectorValuesIterator vectorValuesIterator) {
        this.vectorValuesIterator = vectorValuesIterator;
    }

    public abstract T getVector() throws IOException;

    public int dimension() {
        return dimension;
    }

    public long totalLiveDocs() {
        if (vectorValuesIterator instanceof KNNVectorValuesIterator.DocIdsIteratorValues) {
            DocIdSetIterator docIdSetIterator = vectorValuesIterator.getDocIdSetIterator();
            if (docIdSetIterator instanceof BinaryDocValues) {
                KNNCodecUtil.getTotalLiveDocsCount((BinaryDocValues) docIdSetIterator);
            } else if (docIdSetIterator instanceof FloatVectorValues || docIdSetIterator instanceof ByteVectorValues) {
                return docIdSetIterator.cost();
            }
        } else if (vectorValuesIterator instanceof KNNVectorValuesIterator.FieldWriterIteratorValues) {
            KNNVectorValuesIterator.FieldWriterIteratorValues<?> fieldWriterIteratorValues =
                (KNNVectorValuesIterator.FieldWriterIteratorValues<?>) vectorValuesIterator;
            return fieldWriterIteratorValues.getVectorValue().size();
        }
        // TODO: Rename this exception
        throw new RuntimeException("Not valid KNNVectorValuesIterator for KNNFloatVectorValues, hence not able to " + "find live docs");
    }

    public static class KNNFloatVectorValues extends KNNVectorValues<float[]> {
        public KNNFloatVectorValues(KNNVectorValuesIterator vectorValuesIterator) {
            super(vectorValuesIterator);
        }

        public int docId() {
            return vectorValuesIterator.docId();
        }

        public int advance(int docId) throws IOException {
            return vectorValuesIterator.advance(docId);
        }

        public int nextDoc() throws IOException {
            return vectorValuesIterator.nextDoc();
        }

        @Override
        @SuppressWarnings("unchecked")
        public float[] getVector() throws IOException {
            float[] vector;
            if (vectorValuesIterator instanceof KNNVectorValuesIterator.DocIdsIteratorValues) {
                DocIdSetIterator docIdSetIterator = vectorValuesIterator.getDocIdSetIterator();
                if (docIdSetIterator instanceof BinaryDocValues) {
                    // TODO: See what we want to do with this, this code can be abstracted
                    BinaryDocValues values = (BinaryDocValues) docIdSetIterator;
                    BytesRef bytesref = values.binaryValue();
                    ByteArrayInputStream byteStream = new ByteArrayInputStream(bytesref.bytes, bytesref.offset, bytesref.length);
                    final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                    vector = vectorSerializer.byteToFloatArray(byteStream);
                    dimension = vector.length;
                    return vector;
                } else if (docIdSetIterator instanceof FloatVectorValues) {
                    vector = ((FloatVectorValues) docIdSetIterator).vectorValue();
                    dimension = vector.length;
                    return vector;
                }
            } else if (vectorValuesIterator instanceof KNNVectorValuesIterator.FieldWriterIteratorValues) {
                KNNVectorValuesIterator.FieldWriterIteratorValues<float[]> fieldWriterIteratorValues =
                    (KNNVectorValuesIterator.FieldWriterIteratorValues<float[]>) vectorValuesIterator;
                vector = fieldWriterIteratorValues.getVectorValue().get(docId());
                dimension = vector.length;
                return vector;
            }
            // TODO: Rename this exception
            throw new RuntimeException("Not valid KNNVectorValuesIterator for KNNFloatVectorValues");
        }
    }

    public static class KNNByteVectorValues extends KNNVectorValues<byte[]> {
        public KNNByteVectorValues(KNNVectorValuesIterator vectorValuesIterator) {
            super(vectorValuesIterator);
        }

        public int docId() {
            return vectorValuesIterator.docId();
        }

        public int advance(int docId) throws IOException {
            return vectorValuesIterator.advance(docId);
        }

        public int nextDoc() throws IOException {
            return vectorValuesIterator.nextDoc();
        }

        @Override
        @SuppressWarnings("unchecked")
        public byte[] getVector() throws IOException {
            byte[] vector;
            if (vectorValuesIterator instanceof KNNVectorValuesIterator.DocIdsIteratorValues) {
                DocIdSetIterator docIdSetIterator = vectorValuesIterator.getDocIdSetIterator();
                if (docIdSetIterator instanceof BinaryDocValues) {
                    // TODO: Rename this exception
                    throw new RuntimeException(
                        "Byte vector implementation is not present for BinaryDocValues with "
                            + " using KNNVectorValuesIterator for KNNByteVectorValues"
                    );
                } else if (docIdSetIterator instanceof ByteVectorValues) {
                    return ((ByteVectorValues) docIdSetIterator).vectorValue();
                }

            } else if (vectorValuesIterator instanceof KNNVectorValuesIterator.FieldWriterIteratorValues) {
                KNNVectorValuesIterator.FieldWriterIteratorValues<byte[]> fieldWriterIteratorValues =
                    (KNNVectorValuesIterator.FieldWriterIteratorValues<byte[]>) vectorValuesIterator;
                vector = fieldWriterIteratorValues.getVectorValue().get(docId());
                dimension = vector.length;
                return vector;
            }
            // TODO: Rename this exception
            throw new RuntimeException("Not valid KNNVectorValuesIterator for KNNByteVectorValues");
        }
    }

}
