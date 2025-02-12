/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.quantizationservice;

import org.opensearch.knn.KNNTestCase;
import org.junit.Before;

import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.MultiBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;

public class QuantizationServiceTests extends KNNTestCase {
    private QuantizationService<float[], byte[]> quantizationService;
    private Supplier<KNNVectorValues<float[]>> knnVectorValues;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        quantizationService = QuantizationService.getInstance();

        // Predefined float vectors for testing
        List<float[]> floatVectors = List.of(
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 4.0f, 5.0f, 6.0f },
            new float[] { 7.0f, 8.0f, 9.0f }
        );

        // Use the predefined vectors to create KNNVectorValues
        // Use the predefined vectors to create KNNVectorValues
        knnVectorValues = () -> KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(floatVectors)
        );
    }

    public void testTrain_oneBitQuantizer_success() throws IOException {
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        QuantizationState quantizationState = quantizationService.train(
            oneBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );

        assertTrue(quantizationState instanceof OneBitScalarQuantizationState);
        OneBitScalarQuantizationState oneBitState = (OneBitScalarQuantizationState) quantizationState;

        // Validate the mean thresholds obtained from the training
        float[] thresholds = oneBitState.getMeanThresholds();
        assertNotNull("Thresholds should not be null", thresholds);
        assertEquals("Thresholds array length should match the dimension", 3, thresholds.length);

        // Example expected thresholds based on the provided vectors
        assertArrayEquals(new float[] { 4.0f, 5.0f, 6.0f }, thresholds, 0.1f);
    }

    public void testTrain_twoBitQuantizer_success() throws IOException {
        ScalarQuantizationParams twoBitParams = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        QuantizationState quantizationState = quantizationService.train(
            twoBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );

        assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState multiBitState = (MultiBitScalarQuantizationState) quantizationState;

        // Validate the thresholds obtained from the training
        float[][] thresholds = multiBitState.getThresholds();
        assertNotNull("Thresholds should not be null", thresholds);
        assertEquals("Number of bits should match the number of rows", 2, thresholds.length);
        assertEquals("Thresholds array length should match the dimension", 3, thresholds[0].length);

        // // Example expected thresholds for two-bit quantization
        float[][] expectedThresholds = {
            { 3.1835034f, 4.1835036f, 5.1835036f },  // First bit level
            { 4.816497f, 5.816497f, 6.816497f }   // Second bit level
        };
        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(expectedThresholds[i], thresholds[i], 0.1f);
        }
    }

    public void testTrain_fourBitQuantizer_success() throws IOException {
        ScalarQuantizationParams fourBitParams = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        QuantizationState quantizationState = quantizationService.train(
            fourBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );

        assertTrue(quantizationState instanceof MultiBitScalarQuantizationState);
        MultiBitScalarQuantizationState multiBitState = (MultiBitScalarQuantizationState) quantizationState;

        // Validate the thresholds obtained from the training
        float[][] thresholds = multiBitState.getThresholds();
        assertNotNull("Thresholds should not be null", thresholds);
        assertEquals("Number of bits should match the number of rows", 4, thresholds.length);
        assertEquals("Thresholds array length should match the dimension", 3, thresholds[0].length);

        // // Example expected thresholds for four-bit quantization
        float[][] expectedThresholds = {
            { 2.530306f, 3.530306f, 4.530306f },  // First bit level
            { 3.510102f, 4.5101023f, 5.5101023f },  // Second bit level
            { 4.489898f, 5.489898f, 6.489898f },  // Third bit level
            { 5.469694f, 6.469694f, 7.469694f }   // Fourth bit level
        };
        for (int i = 0; i < thresholds.length; i++) {
            assertArrayEquals(expectedThresholds[i], thresholds[i], 0.1f);
        }
    }

    public void testQuantize_oneBitQuantizer_success() throws IOException {
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        QuantizationState quantizationState = quantizationService.train(
            oneBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );

        QuantizationOutput quantizationOutput = quantizationService.createQuantizationOutput(oneBitParams);

        byte[] quantizedVector = quantizationService.quantize(quantizationState, new float[] { 1.0f, 2.0f, 3.0f }, quantizationOutput);

        assertNotNull("Quantized vector should not be null", quantizedVector);

        // Expected quantized vector values for one-bit quantization (packed bits)
        byte[] expectedQuantizedVector = new byte[] { 0 };  // 00000000 (all bits are 0)
        assertArrayEquals(expectedQuantizedVector, quantizedVector);
    }

    public void testQuantize_twoBitQuantizer_success() throws IOException {
        ScalarQuantizationParams twoBitParams = new ScalarQuantizationParams(ScalarQuantizationType.TWO_BIT);
        QuantizationState quantizationState = quantizationService.train(
            twoBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );
        QuantizationOutput quantizationOutput = quantizationService.createQuantizationOutput(twoBitParams);
        byte[] quantizedVector = quantizationService.quantize(quantizationState, new float[] { 4.0f, 5.0f, 6.0f }, quantizationOutput);

        assertNotNull("Quantized vector should not be null", quantizedVector);

        // Expected quantized vector values for two-bit quantization (packed bits)
        byte[] expectedQuantizedVector = new byte[] { (byte) 0b11100000 };
        assertArrayEquals(expectedQuantizedVector, quantizedVector);
    }

    public void testQuantize_fourBitQuantizer_success() throws IOException {
        ScalarQuantizationParams fourBitParams = new ScalarQuantizationParams(ScalarQuantizationType.FOUR_BIT);
        QuantizationState quantizationState = quantizationService.train(
            fourBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );
        QuantizationOutput quantizationOutput = quantizationService.createQuantizationOutput(fourBitParams);

        byte[] quantizedVector = quantizationService.quantize(quantizationState, new float[] { 7.0f, 8.0f, 9.0f }, quantizationOutput);

        assertNotNull("Quantized vector should not be null", quantizedVector);

        // Expected quantized vector values for four-bit quantization (packed bits)
        byte[] expectedQuantizedVector = new byte[] { (byte) 0xFF, (byte) 0xF0 };
        assertArrayEquals(expectedQuantizedVector, quantizedVector);
    }

    public void testQuantize_whenInvalidInput_thenThrows() throws IOException {
        ScalarQuantizationParams oneBitParams = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        QuantizationState quantizationState = quantizationService.train(
            oneBitParams,
            knnVectorValues,
            knnVectorValues.get().totalLiveDocs()
        );
        QuantizationOutput quantizationOutput = quantizationService.createQuantizationOutput(oneBitParams);
        assertThrows(IllegalArgumentException.class, () -> quantizationService.quantize(quantizationState, null, quantizationOutput));
    }
}
