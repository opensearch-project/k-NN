/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

/**
 * To avoid unit test duplication, tests for exception is added here. For non exception cases tests are present in
 * {@link KNNVectorValuesTests}
 */
public class VectorValueExtractorStrategyTests extends KNNTestCase {

    @SneakyThrows
    public void testExtractWithDISI_whenInvalidIterator_thenException() {
        final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
        final KNNVectorValuesIterator vectorValuesIterator = Mockito.mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        Mockito.when(vectorValuesIterator.getDocIdSetIterator()).thenReturn(new TestVectorValues.NotBinaryDocValues());
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.FLOAT, vectorValuesIterator));
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.HALF_FLOAT, vectorValuesIterator));
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.BINARY, vectorValuesIterator));
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.BYTE, vectorValuesIterator));
    }
}
