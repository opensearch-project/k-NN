/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class SegmentLevelQuantizationInfoTests extends KNNTestCase {

    private static final String FIELD_NAME = "my_vector";

    private QuantizationParams quantizationParams() {
        return ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build();
    }

    /**
     * fieldInfo == null -> return null, and quantization params are never even looked up.
     */
    @SneakyThrows
    public void testBuild_whenFieldInfoIsNull_thenReturnsNull() {
        final LeafReader leafReader = mock(LeafReader.class);
        try (MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class)) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);

            assertNull(SegmentLevelQuantizationInfo.build(leafReader, null, FIELD_NAME));

            // The field-is-vector guard runs before any quantization lookup.
            verify(quantizationService, never()).getQuantizationParams(any());
            verify(leafReader, never()).getFloatVectorValues(anyString());
        }
    }

    /**
     * fieldInfo present but not a vector field -> return null before any quantization lookup.
     */
    @SneakyThrows
    public void testBuild_whenFieldIsNotVectorField_thenReturnsNull() {
        final LeafReader leafReader = mock(LeafReader.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.hasVectorValues()).thenReturn(false);

        try (MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class)) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);

            assertNull(SegmentLevelQuantizationInfo.build(leafReader, fieldInfo, FIELD_NAME));

            verify(quantizationService, never()).getQuantizationParams(any());
            verify(leafReader, never()).getFloatVectorValues(anyString());
        }
    }

    /**
     * Vector field but no quantization params (non-quantized index) -> return null without reading vectors.
     */
    @SneakyThrows
    public void testBuild_whenQuantizationParamsIsNull_thenReturnsNull() {
        final LeafReader leafReader = mock(LeafReader.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.hasVectorValues()).thenReturn(true);

        try (MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class)) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);
            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(null);

            assertNull(SegmentLevelQuantizationInfo.build(leafReader, fieldInfo, FIELD_NAME));

            // No params -> we must not attempt to read the (possibly missing) quantization state.
            verify(leafReader, never()).getFloatVectorValues(anyString());
        }
    }

    /**
     * Core bug fix: quantized field is present in FieldInfos but the segment has zero live vector docs
     * ({@link FloatVectorValues} is null). build() must return null instead of trying to open the
     * missing .osknnqstate file.
     */
    @SneakyThrows
    public void testBuild_whenFloatVectorValuesIsNull_thenReturnsNull() {
        final LeafReader leafReader = mock(LeafReader.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(null);

        try (
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<SegmentLevelQuantizationUtil> utilMockedStatic = mockStatic(SegmentLevelQuantizationUtil.class)
        ) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);
            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams());

            assertNull(SegmentLevelQuantizationInfo.build(leafReader, fieldInfo, FIELD_NAME));

            // Reader must not be asked for a quantization state that was never written.
            utilMockedStatic.verify(() -> SegmentLevelQuantizationUtil.getQuantizationState(any(), anyString()), never());
        }
    }

    /**
     * Core bug fix: quantized field present but zero live vector docs (size() == 0) -> return null.
     */
    @SneakyThrows
    public void testBuild_whenFloatVectorValuesIsEmpty_thenReturnsNull() {
        final LeafReader leafReader = mock(LeafReader.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        final FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        when(floatVectorValues.size()).thenReturn(0);
        when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(floatVectorValues);

        try (
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<SegmentLevelQuantizationUtil> utilMockedStatic = mockStatic(SegmentLevelQuantizationUtil.class)
        ) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);
            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams());

            assertNull(SegmentLevelQuantizationInfo.build(leafReader, fieldInfo, FIELD_NAME));

            utilMockedStatic.verify(() -> SegmentLevelQuantizationUtil.getQuantizationState(any(), anyString()), never());
        }
    }

    /**
     * Happy path: quantized field with live vector docs -> build the info carrying the params and state.
     */
    @SneakyThrows
    public void testBuild_whenFieldHasVectors_thenReturnsInfo() {
        final LeafReader leafReader = mock(LeafReader.class);
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.hasVectorValues()).thenReturn(true);
        final FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        when(floatVectorValues.size()).thenReturn(5);
        when(leafReader.getFloatVectorValues(FIELD_NAME)).thenReturn(floatVectorValues);

        final QuantizationParams quantizationParams = quantizationParams();
        final QuantizationState quantizationState = OneBitScalarQuantizationState.builder()
            .quantizationParams((ScalarQuantizationParams) quantizationParams)
            .meanThresholds(new float[] { 1.2f, 2.3f, 3.4f, 4.5f })
            .build();

        try (
            MockedStatic<QuantizationService> quantizationServiceMockedStatic = mockStatic(QuantizationService.class);
            MockedStatic<SegmentLevelQuantizationUtil> utilMockedStatic = mockStatic(SegmentLevelQuantizationUtil.class)
        ) {
            final QuantizationService quantizationService = mock(QuantizationService.class);
            quantizationServiceMockedStatic.when(QuantizationService::getInstance).thenReturn(quantizationService);
            when(quantizationService.getQuantizationParams(fieldInfo)).thenReturn(quantizationParams);
            utilMockedStatic.when(() -> SegmentLevelQuantizationUtil.getQuantizationState(leafReader, FIELD_NAME))
                .thenReturn(quantizationState);

            final SegmentLevelQuantizationInfo info = SegmentLevelQuantizationInfo.build(leafReader, fieldInfo, FIELD_NAME);

            assertNotNull(info);
            assertEquals(quantizationParams, info.getQuantizationParams());
            assertEquals(quantizationState, info.getQuantizationState());
            utilMockedStatic.verify(() -> SegmentLevelQuantizationUtil.getQuantizationState(leafReader, FIELD_NAME));
        }
    }
}
