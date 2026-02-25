/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

public class QuantizedKNNBinaryVectorValuesTests extends KNNTestCase {
    @Test
    public void testQuantizedKNNBinaryVectorValues() throws IOException {
        // Prepare random float vectors
        final int numberOfVectors = 500;
        final int dimension = 40;
        List<float[]> floatVectors = Arrays.asList(TestVectorValues.getRandomVectors(numberOfVectors, dimension));

        // Prepare quantization
        final QuantizationService quantizationService = QuantizationService.getInstance();
        final ScalarQuantizationParams quantizationParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT, false, false);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(floatVectors)
        );

        // Do quantization
        final QuantizationState quantizationState = quantizationService.train(quantizationParams, knnVectorValuesSupplier, numberOfVectors);

        // Create QuantizedKNNBinaryVectorValues
        final BuildIndexParams mockedParams = mock(BuildIndexParams.class);
        when(mockedParams.getQuantizationState()).thenReturn(quantizationState);
        final QuantizedKNNBinaryVectorValues quantizingVectorValues = new QuantizedKNNBinaryVectorValues(
            knnVectorValuesSupplier.get(),
            mockedParams
        );

        // Compare quantized bytes
        initializeVectorValues(quantizingVectorValues);
        assertEquals(dimension, quantizingVectorValues.dimension());
        // 40 vectors will be quantized into 40 bits (we're applying 32x compression here)
        // therefore, 40 / 8 bytes = 5 is expected
        assertEquals(5, quantizingVectorValues.bytesPerVector());

        for (int i = 0; i < numberOfVectors; i++) {
            // We should get correctly quantized byte[]
            final byte[] expectedBytes = (byte[]) quantizationService.quantize(
                quantizationState,
                floatVectors.get(i),
                quantizationService.createQuantizationOutput(quantizationParams)
            );

            final byte[] acquiredBytes = quantizingVectorValues.getVector();

            assertArrayEquals(expectedBytes, acquiredBytes);

            // Advance to the next vector
            final int nextDocId = quantizingVectorValues.nextDoc();
            if (i == (numberOfVectors - 1)) {
                // At the last, we should get 'no more docs'
                assertEquals(NO_MORE_DOCS, nextDocId);
            } else {
                assertNotEquals(NO_MORE_DOCS, nextDocId);
            }
        }
    }
}
