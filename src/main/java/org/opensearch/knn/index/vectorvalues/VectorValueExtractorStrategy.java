/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.IOException;

/**
 * Provides different strategies to extract the vectors from different {@link KNNVectorValuesIterator}
 */
interface VectorValueExtractorStrategy {

    /**
     * Extract a float vector from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return float[]
     * @throws IOException exception while retrieving the vectors
     */
    static float[] extractFloatVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.FLOAT, iterator);
    }

    /**
     * Extract a byte vector from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return byte[]
     * @throws IOException exception while retrieving the vectors
     */
    static byte[] extractByteVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.BYTE, iterator);
    }

    /**
     * Extract a binary vector which is represented as byte[] from KNNVectorValuesIterator.
     * @param iterator {@link KNNVectorValuesIterator}
     * @return byte[]
     * @throws IOException exception while retrieving the vectors
     */
    static byte[] extractBinaryVector(final KNNVectorValuesIterator iterator) throws IOException {
        return iterator.getVectorExtractorStrategy().extract(VectorDataType.BINARY, iterator);
    }

    /**
     * Extract Vector based on the vector datatype and vector values iterator.
     * @param vectorDataType {@link VectorDataType}
     * @param vectorValuesIterator {@link KNNVectorValuesIterator}
     * @return vector
     * @param <T> could be of type float[], byte[]
     * @throws IOException exception during extracting the vectors
     */
    <T> T extract(VectorDataType vectorDataType, KNNVectorValuesIterator vectorValuesIterator) throws IOException;

    /**
     * Strategy to extract the vector from {@link KNNVectorValuesIterator.DocIdsIteratorValues}
     */
    class DISIVectorExtractor implements VectorValueExtractorStrategy {
        @Override
        public <T> T extract(final VectorDataType vectorDataType, final KNNVectorValuesIterator vectorValuesIterator) throws IOException {
            final DocIdSetIterator docIdSetIterator = vectorValuesIterator.getDocIdSetIterator();
            switch (vectorDataType) {
                case FLOAT:
                    if (docIdSetIterator instanceof BinaryDocValues) {
                        final BinaryDocValues values = (BinaryDocValues) docIdSetIterator;
                        return (T) getFloatVectorFromByteRef(values.binaryValue());
                    } else if (docIdSetIterator instanceof FloatVectorValues) {
                        return (T) ((FloatVectorValues) docIdSetIterator).vectorValue();
                    }
                    throw new IllegalArgumentException(
                        "VectorValuesIterator is not of a valid type. Valid Types are: BinaryDocValues and FloatVectorValues"
                    );
                case BYTE:
                case BINARY:
                    if (docIdSetIterator instanceof BinaryDocValues) {
                        final BinaryDocValues values = (BinaryDocValues) docIdSetIterator;
                        final BytesRef bytesRef = values.binaryValue();
                        return (T) ArrayUtil.copyOfSubArray(bytesRef.bytes, bytesRef.offset, bytesRef.offset + bytesRef.length);
                    } else if (docIdSetIterator instanceof ByteVectorValues) {
                        return (T) ((ByteVectorValues) docIdSetIterator).vectorValue();
                    }
                    throw new IllegalArgumentException(
                        "VectorValuesIterator is not of a valid type. Valid Types are: BinaryDocValues and ByteVectorValues"
                    );
            }
            throw new IllegalArgumentException("Valid Vector data type not passed to extract vector from DISIVectorExtractor strategy");
        }

        private float[] getFloatVectorFromByteRef(final BytesRef bytesRef) {
            final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByBytesRef(bytesRef);
            return vectorSerializer.byteToFloatArray(bytesRef);
        }
    }

    /**
     * Strategy to extract the vector from {@link KNNVectorValuesIterator.FieldWriterIteratorValues}
     */
    class FieldWriterIteratorVectorExtractor implements VectorValueExtractorStrategy {

        @SuppressWarnings("unchecked")
        @Override
        public <T> T extract(final VectorDataType vectorDataType, final KNNVectorValuesIterator vectorValuesIterator) throws IOException {
            switch (vectorDataType) {
                case FLOAT:
                    return (T) ((KNNVectorValuesIterator.FieldWriterIteratorValues<float[]>) vectorValuesIterator).vectorsValue();
                case BYTE:
                case BINARY:
                    return (T) ((KNNVectorValuesIterator.FieldWriterIteratorValues<byte[]>) vectorValuesIterator).vectorsValue();
            }
            throw new IllegalArgumentException(
                "Valid Vector data type not passed to extract vector from FieldWriterIteratorVectorExtractor strategy"
            );
        }
    }

    /**
     * Strategy to extract the vector from {@link KNNVectorValuesIterator.MergeSegmentVectorValuesIterator}
     */
    class MergeSegmentValuesExtractor implements VectorValueExtractorStrategy {
        @Override
        public <T> T extract(final VectorDataType vectorDataType, final KNNVectorValuesIterator vectorValuesIterator) throws IOException {
            switch (vectorDataType) {
                case FLOAT:
                    return (T) ((KNNVectorValuesIterator.MergeFloat32VectorValuesIterator) vectorValuesIterator).vectorValue();
                case BYTE:
                case BINARY:
                    return (T) ((KNNVectorValuesIterator.MergeByteVectorValuesIterator) vectorValuesIterator).vectorValue();
            }
            throw new IllegalArgumentException(
                "Valid Vector data type not passed to extract vector from FieldWriterIteratorVectorExtractor strategy"
            );
        }
    }

}
