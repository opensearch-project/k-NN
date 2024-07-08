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

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEnginesKNNVectorsWriter;

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
}
