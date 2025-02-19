/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class KNNVectorValuesTests extends KNNTestCase {

    @SneakyThrows
    public void testFloatVectorValues_whenValidInput_thenSuccess() {
        final List<float[]> floatArray = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final int dimension = floatArray.get(0).length;
        final TestVectorValues.PreDefinedFloatVectorValues randomVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );
        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, randomVectorValues);
        new CompareVectorValues<float[]>().validateVectorValues(knnVectorValues, floatArray, 8, dimension, true);

        final DocsWithFieldSet docsWithFieldSet = getDocIdSetIterator(floatArray.size());

        final Map<Integer, float[]> vectorsMap = Map.of(0, floatArray.get(0), 1, floatArray.get(1));
        final KNNVectorValues<float[]> knnVectorValuesForFieldWriter = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            vectorsMap
        );
        new CompareVectorValues<float[]>().validateVectorValues(knnVectorValuesForFieldWriter, floatArray, 8, dimension, false);
        final TestVectorValues.PredefinedFloatVectorBinaryDocValues preDefinedFloatVectorValues =
            new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArray);
        final KNNVectorValues<float[]> knnFloatVectorValuesBinaryDocValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            preDefinedFloatVectorValues
        );
        new CompareVectorValues<float[]>().validateVectorValues(knnFloatVectorValuesBinaryDocValues, floatArray, 8, dimension, false);
    }

    @SneakyThrows
    public void testByteVectorValues_whenValidInput_thenSuccess() {
        final List<byte[]> byteArray = List.of(new byte[] { 4, 5 }, new byte[] { 6, 7 });
        final int dimension = byteArray.get(0).length;
        final TestVectorValues.PreDefinedByteVectorValues randomVectorValues = new TestVectorValues.PreDefinedByteVectorValues(byteArray);
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, randomVectorValues);
        new CompareVectorValues<byte[]>().validateVectorValues(knnVectorValues, byteArray, 2, dimension, true);

        final DocsWithFieldSet docsWithFieldSet = getDocIdSetIterator(byteArray.size());
        final Map<Integer, byte[]> vectorsMap = Map.of(0, byteArray.get(0), 1, byteArray.get(1));
        final KNNVectorValues<byte[]> knnVectorValuesForFieldWriter = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            docsWithFieldSet,
            vectorsMap
        );
        new CompareVectorValues<byte[]>().validateVectorValues(knnVectorValuesForFieldWriter, byteArray, 2, dimension, false);

        final TestVectorValues.PredefinedByteVectorBinaryDocValues preDefinedByteVectorValues =
            new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArray);
        final KNNVectorValues<byte[]> knnBinaryVectorValuesBinaryDocValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            preDefinedByteVectorValues
        );
        new CompareVectorValues<byte[]>().validateVectorValues(knnBinaryVectorValuesBinaryDocValues, byteArray, 2, dimension, false);
    }

    @SneakyThrows
    public void testBinaryVectorValues_whenValidInput_thenSuccess() {
        final List<byte[]> byteArray = List.of(new byte[] { 1, 5, 8 }, new byte[] { 6, 7, 9 });
        int dimension = byteArray.get(0).length * 8;
        final TestVectorValues.PreDefinedBinaryVectorValues randomVectorValues = new TestVectorValues.PreDefinedBinaryVectorValues(
            byteArray
        );
        final KNNVectorValues<byte[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BINARY, randomVectorValues);
        new CompareVectorValues<byte[]>().validateVectorValues(knnVectorValues, byteArray, 3, dimension, true);

        final DocsWithFieldSet docsWithFieldSet = getDocIdSetIterator(byteArray.size());
        final Map<Integer, byte[]> vectorsMap = Map.of(0, byteArray.get(0), 1, byteArray.get(1));
        final KNNBinaryVectorValues knnVectorValuesForFieldWriter = (KNNBinaryVectorValues) KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BINARY,
            docsWithFieldSet,
            vectorsMap
        );
        new CompareVectorValues<byte[]>().validateVectorValues(knnVectorValuesForFieldWriter, byteArray, 3, dimension, false);

        final TestVectorValues.PredefinedByteVectorBinaryDocValues preDefinedByteVectorValues =
            new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArray);
        final KNNVectorValues<byte[]> knnBinaryVectorValuesBinaryDocValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BINARY,
            preDefinedByteVectorValues
        );
        new CompareVectorValues<byte[]>().validateVectorValues(knnBinaryVectorValuesBinaryDocValues, byteArray, 3, dimension, false);
    }

    private DocsWithFieldSet getDocIdSetIterator(int numberOfDocIds) {
        final DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        for (int i = 0; i < numberOfDocIds; i++) {
            docsWithFieldSet.add(i);
        }
        return docsWithFieldSet;
    }

    private class CompareVectorValues<T> {
        void validateVectorValues(
            KNNVectorValues<T> vectorValues,
            List<T> vectors,
            int bytesPerVector,
            int dimension,
            boolean validateAddress
        ) throws IOException {
            assertEquals(vectorValues.totalLiveDocs(), vectors.size());
            int docId, i = 0;
            T oldActual = null;
            int oldDocId = -1;
            final KNNVectorValuesIterator iterator = vectorValues.vectorValuesIterator;
            for (docId = iterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS && i < vectors.size(); docId = iterator.nextDoc()) {
                T actual = vectorValues.getVector();
                T clone = vectorValues.conditionalCloneVector();
                T expected = vectors.get(i);
                assertNotEquals(oldDocId, docId);
                assertEquals(dimension, vectorValues.dimension());
                // this will check if reference is correct for the vectors. This is mainly required because for
                // VectorValues of Lucene when reading vectors put the vector at same reference
                if (oldActual != null && validateAddress) {
                    assertSame(actual, oldActual);
                    assertNotSame(clone, oldActual);
                }

                oldActual = actual;
                // this will do the deep equals
                assertArrayEquals(new Object[] { actual }, new Object[] { expected });
                assertArrayEquals(new Object[] { clone }, new Object[] { expected });
                i++;
            }
            assertEquals(bytesPerVector, vectorValues.bytesPerVector);
        }
    }

}
