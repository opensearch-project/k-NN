/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import junit.framework.TestCase;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class KNNIndexShardUnitTests extends TestCase {
    /**
     * Unit test for the determineVectorDataType helper method
     */
    public void testDetermineVectorDataType() {
        Version testVersion = Version.LATEST;

        // ----- Test Case 1: Empty Quantization Config with default FLOAT -----
        // Setup mocks
        FieldInfo fieldInfo1 = Mockito.mock(FieldInfo.class);
        Map<String, String> attributes1 = new HashMap<>(); // No vector data type specified
        Mockito.when(fieldInfo1.attributes()).thenReturn(attributes1);

        QuantizationParams quantizationParams1 = Mockito.mock(QuantizationParams.class);

        // Setup FieldInfoExtractor mock behavior using mockStatic
        try (MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class)) {
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo1, testVersion))
                .thenReturn(QuantizationConfig.EMPTY);
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.determineVectorDataType(fieldInfo1, quantizationParams1, testVersion))
                .thenCallRealMethod();

            // Directly call the method
            VectorDataType result1 = FieldInfoExtractor.determineVectorDataType(fieldInfo1, quantizationParams1, testVersion);

            // Verify default FLOAT is returned
            assertEquals(VectorDataType.FLOAT, result1);
        }

        // ----- Test Case 2: Empty Quantization Config with BINARY specified in attributes -----
        FieldInfo fieldInfo2 = Mockito.mock(FieldInfo.class);
        Map<String, String> attributes2 = new HashMap<>();
        attributes2.put(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue());
        Mockito.when(fieldInfo2.attributes()).thenReturn(attributes2);

        QuantizationParams quantizationParams2 = Mockito.mock(QuantizationParams.class);

        try (MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class)) {
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo2, testVersion))
                .thenReturn(QuantizationConfig.EMPTY);
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.determineVectorDataType(fieldInfo2, quantizationParams2, testVersion))
                .thenCallRealMethod();

            // Directly call the method
            VectorDataType result2 = FieldInfoExtractor.determineVectorDataType(fieldInfo2, quantizationParams2, testVersion);

            // Verify BINARY is returned based on attributes
            assertEquals(VectorDataType.BINARY, result2);
        }

        // ----- Test Case 3: Non-empty Quantization Config with ADC enabled -----
        FieldInfo fieldInfo3 = Mockito.mock(FieldInfo.class);
        QuantizationParams quantizationParams3 = Mockito.mock(QuantizationParams.class);

        try (
            MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<SegmentLevelQuantizationUtil> segmentLevelQuantUtilMock = Mockito.mockStatic(SegmentLevelQuantizationUtil.class)
        ) {

            // Setup for non-empty quantization config
            QuantizationConfig nonEmptyConfig = QuantizationConfig.builder().build();

            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo3, testVersion))
                .thenReturn(nonEmptyConfig);
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.determineVectorDataType(fieldInfo3, quantizationParams3, testVersion))
                .thenCallRealMethod();

            // Setup for ADC enabled
            segmentLevelQuantUtilMock.when(() -> SegmentLevelQuantizationUtil.isAdcEnabled(quantizationParams3)).thenReturn(true);

            // Directly call the method
            VectorDataType result3 = FieldInfoExtractor.determineVectorDataType(fieldInfo3, quantizationParams3, testVersion);

            // Verify FLOAT is returned when ADC is enabled
            assertEquals(VectorDataType.FLOAT, result3);
        }

        // ----- Test Case 4: Non-empty Quantization Config with ADC disabled -----
        FieldInfo fieldInfo4 = Mockito.mock(FieldInfo.class);
        QuantizationParams quantizationParams4 = Mockito.mock(QuantizationParams.class);

        try (
            MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<SegmentLevelQuantizationUtil> segmentLevelQuantUtilMock = Mockito.mockStatic(SegmentLevelQuantizationUtil.class)
        ) {

            // Setup for non-empty quantization config
            QuantizationConfig nonEmptyConfig = QuantizationConfig.builder().build();

            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo4, testVersion))
                .thenReturn(nonEmptyConfig);
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.determineVectorDataType(fieldInfo4, quantizationParams4, testVersion))
                .thenCallRealMethod();

            // Setup for ADC disabled
            segmentLevelQuantUtilMock.when(() -> SegmentLevelQuantizationUtil.isAdcEnabled(quantizationParams4)).thenReturn(false);

            // Directly call the method
            VectorDataType result4 = FieldInfoExtractor.determineVectorDataType(fieldInfo4, quantizationParams4, testVersion);

            // Verify BINARY is returned when ADC is disabled
            assertEquals(VectorDataType.BINARY, result4);
        }
    }
}
