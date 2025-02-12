/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.junit.Before;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class QuantizationIndexUtilsTests extends KNNTestCase {

    private KNNVectorValues<float[]> knnVectorValues;
    private BuildIndexParams buildIndexParams;
    private QuantizationService<float[], byte[]> quantizationService;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        quantizationService = mock(QuantizationService.class);

        // Predefined float vectors for testing
        List<float[]> floatVectors = List.of(
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 4.0f, 5.0f, 6.0f },
            new float[] { 7.0f, 8.0f, 9.0f }
        );

        // Use the predefined vectors to create KNNVectorValues
        knnVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(floatVectors)
        );

        // Mocking BuildIndexParams
        buildIndexParams = mock(BuildIndexParams.class);
    }

    public void testPrepareIndexBuild_withQuantization_success() {
        QuantizationState quantizationState = mock(OneBitScalarQuantizationState.class);
        QuantizationOutput quantizationOutput = mock(QuantizationOutput.class);

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        when(quantizationOutput.getQuantizedVector()).thenReturn(new byte[] { 0x01 });
        when(quantizationState.getDimensions()).thenReturn(2);
        when(quantizationState.getBytesPerVector()).thenReturn(8);
        when(quantizationState.getQuantizationParams()).thenReturn(params);

        when(buildIndexParams.getQuantizationState()).thenReturn(quantizationState);

        IndexBuildSetup setup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, buildIndexParams);

        assertNotNull(setup.getQuantizationState());
        assertEquals(8, setup.getBytesPerVector());
        assertEquals(2, setup.getDimensions());
    }

    public void testPrepareIndexBuild_withoutQuantization_success() throws IOException {
        when(buildIndexParams.getQuantizationState()).thenReturn(null);
        knnVectorValues.nextDoc();
        knnVectorValues.getVector();
        IndexBuildSetup setup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, buildIndexParams);
        assertNull(setup.getQuantizationState());
        assertEquals(knnVectorValues.bytesPerVector(), setup.getBytesPerVector());
        assertEquals(knnVectorValues.dimension(), setup.getDimensions());
    }

    public void testProcessAndReturnVector_withoutQuantization_success() throws IOException {
        // Set up the BuildIndexParams to return no quantization
        when(buildIndexParams.getQuantizationState()).thenReturn(null);
        knnVectorValues.nextDoc();
        knnVectorValues.getVector();
        IndexBuildSetup setup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, buildIndexParams);
        // Process and return the vector
        assertNotNull(QuantizationIndexUtils.processAndReturnVector(knnVectorValues, setup));
    }

    public void testProcessAndReturnVector_withQuantization_success() throws IOException {
        // Set up quantization state and output
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        float[] mean = { 1.0f, 2.0f, 3.0f };
        knnVectorValues.nextDoc();
        OneBitScalarQuantizationState state = OneBitScalarQuantizationState.builder()
            .quantizationParams(params)
            .meanThresholds(mean)
            .build();
        QuantizationOutput quantizationOutput = mock(QuantizationOutput.class);
        when(buildIndexParams.getQuantizationState()).thenReturn(state);
        IndexBuildSetup setup = QuantizationIndexUtils.prepareIndexBuild(knnVectorValues, buildIndexParams);
        // Process and return the vector
        Object result = QuantizationIndexUtils.processAndReturnVector(knnVectorValues, setup);
        assertTrue(result instanceof byte[]);
        assertArrayEquals(new byte[] { 0x00 }, (byte[]) result);
    }
}
