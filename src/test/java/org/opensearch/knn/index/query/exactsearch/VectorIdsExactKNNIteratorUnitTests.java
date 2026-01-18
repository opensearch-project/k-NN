/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.exactsearch;

import junit.framework.TestCase;
import org.apache.lucene.search.DocIdSetIterator;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

import static org.junit.Assert.assertThrows;;

public class VectorIdsExactKNNIteratorUnitTests extends TestCase {
    @Mock
    private DocIdSetIterator mockFilterIdsIterator;
    @Mock
    private KNNFloatVectorValues mockKnnFloatVectorValues;

    private VectorIdsExactKNNIterator iterator;
    private float[] queryVector;
    private SpaceType spaceType;

    @Before
    public void setUp() throws IOException {
        mockFilterIdsIterator = Mockito.mock(DocIdSetIterator.class);
        mockKnnFloatVectorValues = Mockito.mock(KNNFloatVectorValues.class);
        queryVector = new float[] { 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        spaceType = SpaceType.L2;

        // Initialize the iterator with mocked dependencies
        iterator = new VectorIdsExactKNNIterator(mockFilterIdsIterator, queryVector, mockKnnFloatVectorValues, spaceType);
    }

    public void testShouldScoreWithADC() {
        // Test case 1: null segmentLevelQuantizationInfo
        assertFalse(iterator.shouldScoreWithADC(null));

        // Test case 2: ScalarQuantizationParams with ADC enabled
        SegmentLevelQuantizationInfo segmentInfoEnabled = Mockito.mock(SegmentLevelQuantizationInfo.class);
        ScalarQuantizationParams scalarParamsEnabled = Mockito.mock(ScalarQuantizationParams.class);
        Mockito.when(segmentInfoEnabled.getQuantizationParams()).thenReturn(scalarParamsEnabled);
        Mockito.when(scalarParamsEnabled.isEnableADC()).thenReturn(true);
        assertTrue(iterator.shouldScoreWithADC(segmentInfoEnabled));

        // Test case 3: ScalarQuantizationParams with ADC disabled
        SegmentLevelQuantizationInfo segmentInfoDisabled = Mockito.mock(SegmentLevelQuantizationInfo.class);
        ScalarQuantizationParams scalarParamsDisabled = Mockito.mock(ScalarQuantizationParams.class);
        Mockito.when(segmentInfoDisabled.getQuantizationParams()).thenReturn(scalarParamsDisabled);
        Mockito.when(scalarParamsDisabled.isEnableADC()).thenReturn(false);
        assertFalse(iterator.shouldScoreWithADC(segmentInfoDisabled));

        // Test case 4: Non-ScalarQuantizationParams
        SegmentLevelQuantizationInfo segmentInfoNonScalar = Mockito.mock(SegmentLevelQuantizationInfo.class);
        QuantizationParams nonScalarParams = Mockito.mock(QuantizationParams.class);
        Mockito.when(segmentInfoNonScalar.getQuantizationParams()).thenReturn(nonScalarParams);
        assertFalse(iterator.shouldScoreWithADC(segmentInfoNonScalar));
    }

    public void testScoreWithADC() {
        SegmentLevelQuantizationInfo mockInfo = Mockito.mock(SegmentLevelQuantizationInfo.class);
        byte[] documentVector = new byte[] { 0 };

        // Test case 1: L2 space type
        try (MockedStatic<KNNScoringUtil> knnScoringUtilMock = Mockito.mockStatic(KNNScoringUtil.class)) {
            knnScoringUtilMock.when(() -> KNNScoringUtil.l2SquaredADC(queryVector, documentVector)).thenReturn(5.0f);

            float result = iterator.scoreWithADC(queryVector, documentVector, SpaceType.L2);
            assertEquals(SpaceType.L2.scoreTranslation(5.0f), result);
        }

        // Test case 2: INNER_PRODUCT space type
        try (MockedStatic<KNNScoringUtil> knnScoringUtilMock = Mockito.mockStatic(KNNScoringUtil.class)) {
            knnScoringUtilMock.when(() -> KNNScoringUtil.innerProductADC(queryVector, documentVector)).thenReturn(10.0f);

            float result = iterator.scoreWithADC(queryVector, documentVector, SpaceType.INNER_PRODUCT);
            assertEquals(SpaceType.INNER_PRODUCT.scoreTranslation(-10.0f), result);
        }

        // Test case 3: COSINESIMIL space type
        try (MockedStatic<KNNScoringUtil> knnScoringUtilMock = Mockito.mockStatic(KNNScoringUtil.class)) {
            knnScoringUtilMock.when(() -> KNNScoringUtil.innerProductADC(queryVector, documentVector)).thenReturn(15.0f);

            float result = iterator.scoreWithADC(queryVector, documentVector, SpaceType.COSINESIMIL);
            assertEquals(SpaceType.COSINESIMIL.scoreTranslation(1 - 15.0f), result);
        }

        // Test case 4: Unsupported space type - using a different existing enum value
        // that isn't handled in the method
        UnsupportedOperationException exception = assertThrows(
            UnsupportedOperationException.class,
            () -> iterator.scoreWithADC(queryVector, documentVector, SpaceType.HAMMING)
        );
        assertEquals("Space type " + SpaceType.HAMMING.getValue() + " is not supported for ADC", exception.getMessage());
    }
}
