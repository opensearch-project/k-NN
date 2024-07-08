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

import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEnginesKNNVectorsWriter;

import java.io.IOException;
import java.util.List;

/**
 * This is a factory class that provides various methods to create the {@link KNNVectorValues}
 */
public class KNNVectorValuesFactory {

    public static KNNVectorValues<float[]> getFloatVectorValues(final DocIdSetIterator docIdSetIterator) {
        KNNVectorValuesIterator vectorValuesIterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator);
        return new KNNVectorValues.KNNFloatVectorValues(vectorValuesIterator);
    }

    public static KNNVectorValues<byte[]> getByteVectorValues(final DocIdSetIterator docIdSetIterator) {
        KNNVectorValuesIterator vectorValuesIterator = new KNNVectorValuesIterator.DocIdsIteratorValues(docIdSetIterator);
        return new KNNVectorValues.KNNByteVectorValues(vectorValuesIterator);
    }

    public static KNNVectorValues<float[]> getFloatVectorValues(final NativeEnginesKNNVectorsWriter.FieldWriter<?> fieldWriter) {
        KNNVectorValuesIterator vectorValuesIterator =
                new KNNVectorValuesIterator.FieldWriterIteratorValues<>(fieldWriter.getDocsWithField().iterator(),
                fieldWriter.getVectors().iterator());
        return new KNNVectorValues.KNNFloatVectorValues(vectorValuesIterator);
    }

    public static KNNVectorValues<byte[]> getByteVectorValues(final NativeEnginesKNNVectorsWriter.FieldWriter<?> fieldWriter) {
        KNNVectorValuesIterator vectorValuesIterator =
                new KNNVectorValuesIterator.FieldWriterIteratorValues<>(fieldWriter.getDocsWithField().iterator(),
                        fieldWriter.getVectors().iterator());
        return new KNNVectorValues.KNNByteVectorValues(vectorValuesIterator);
    }

    public static KNNVectorValues<float[]> getFloatVectorValues(final LeafReaderContext leafReaderContext, String field)
        throws IOException {
        final SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(leafReaderContext.reader());
        DocIdSetIterator docIdSetIterator;
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(field);
        if (fieldInfo.getVectorDimension() > 0) {
            docIdSetIterator = reader.getFloatVectorValues(field);
        } else {
            docIdSetIterator = DocValues.getBinary(leafReaderContext.reader(), field);
        }
        return getFloatVectorValues(docIdSetIterator);
    }

    public static KNNVectorValues<byte[]> getByteVectorValues(final LeafReaderContext leafReaderContext, String field) throws IOException {
        final SegmentReader reader = (SegmentReader) FilterLeafReader.unwrap(leafReaderContext.reader());
        DocIdSetIterator docIdSetIterator;
        final FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(field);
        if (fieldInfo.getVectorDimension() > 0) {
            docIdSetIterator = reader.getByteVectorValues(field);
        } else {
            docIdSetIterator = DocValues.getBinary(leafReaderContext.reader(), field);
        }
        return getByteVectorValues(docIdSetIterator);
    }
}
