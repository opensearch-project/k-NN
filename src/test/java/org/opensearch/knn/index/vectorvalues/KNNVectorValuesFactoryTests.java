/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.junit.Assert;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import java.util.Map;

public class KNNVectorValuesFactoryTests extends KNNTestCase {
    private static final int COUNT = 10;
    private static final int DIMENSION = 10;

    public void testGetVectorValuesFromDISI_whenValidInput_thenSuccess() {
        final BinaryDocValues binaryDocValues = new TestVectorValues.RandomVectorBinaryDocValues(COUNT, DIMENSION);
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, binaryDocValues);
        Assert.assertNotNull(floatVectorValues);

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, binaryDocValues);
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BINARY, binaryDocValues);
        Assert.assertNotNull(binaryVectorValues);
    }

    public void testGetVectorValuesUsingDocWithFieldSet_whenValidInput_thenSuccess() {
        final DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docsWithFieldSet.add(0);
        docsWithFieldSet.add(1);
        final Map<Integer, float[]> floatVectorMap = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 });
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            floatVectorMap
        );
        Assert.assertNotNull(floatVectorValues);

        final Map<Integer, byte[]> byteVectorMap = Map.of(0, new byte[] { 4, 5 }, 1, new byte[] { 6, 7 });

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BINARY,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(binaryVectorValues);
    }

}
