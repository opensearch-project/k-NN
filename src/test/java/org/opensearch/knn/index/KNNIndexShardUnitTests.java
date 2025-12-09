/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class KNNIndexShardUnitTests extends KNNTestCase {
    /**
     * Unit test for the determineVectorDataType helper method
     */
    public void testDetermineVectorDataType() {
        KNNIndexShard knnIndexShard = new KNNIndexShard(null);
        Version testVersion = Version.LATEST;

        // ----- Test Case 1: Empty Quantization Config with default FLOAT -----
        // Setup mocks
        FieldInfo fieldInfo1 = Mockito.mock(FieldInfo.class);
        Map<String, String> attributes1 = new HashMap<>(); // No vector data type specified
        Mockito.when(fieldInfo1.attributes()).thenReturn(attributes1);

        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo1 = Mockito.mock(SegmentLevelQuantizationInfo.class);

        // Setup FieldInfoExtractor mock behavior using mockStatic
        try (MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class)) {
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo1, testVersion))
                .thenReturn(QuantizationConfig.EMPTY);

            // Directly call the method
            VectorDataType result1 = knnIndexShard.determineVectorDataType(fieldInfo1, segmentLevelQuantizationInfo1, testVersion);

            // Verify default FLOAT is returned
            assertEquals(VectorDataType.FLOAT, result1);
        }

        // ----- Test Case 2: Empty Quantization Config with BINARY specified in attributes -----
        FieldInfo fieldInfo2 = Mockito.mock(FieldInfo.class);
        Map<String, String> attributes2 = new HashMap<>();
        attributes2.put(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue());
        Mockito.when(fieldInfo2.attributes()).thenReturn(attributes2);

        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo2 = Mockito.mock(SegmentLevelQuantizationInfo.class);

        try (MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class)) {
            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo2, testVersion))
                .thenReturn(QuantizationConfig.EMPTY);

            // Directly call the method
            VectorDataType result2 = knnIndexShard.determineVectorDataType(fieldInfo2, segmentLevelQuantizationInfo2, testVersion);

            // Verify BINARY is returned based on attributes
            assertEquals(VectorDataType.BINARY, result2);
        }

        // ----- Test Case 3: Non-empty Quantization Config with ADC enabled -----
        FieldInfo fieldInfo3 = Mockito.mock(FieldInfo.class);
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo3 = Mockito.mock(SegmentLevelQuantizationInfo.class);

        try (
            MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<SegmentLevelQuantizationUtil> segmentLevelQuantUtilMock = Mockito.mockStatic(SegmentLevelQuantizationUtil.class)
        ) {

            // Setup for non-empty quantization config
            QuantizationConfig nonEmptyConfig = QuantizationConfig.builder().build();

            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo3, testVersion))
                .thenReturn(nonEmptyConfig);

            // Setup for ADC enabled
            segmentLevelQuantUtilMock.when(() -> SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo3)).thenReturn(true);

            // Directly call the method
            VectorDataType result3 = knnIndexShard.determineVectorDataType(fieldInfo3, segmentLevelQuantizationInfo3, testVersion);

            // Verify FLOAT is returned when ADC is enabled
            assertEquals(VectorDataType.FLOAT, result3);
        }

        // ----- Test Case 4: Non-empty Quantization Config with ADC disabled -----
        FieldInfo fieldInfo4 = Mockito.mock(FieldInfo.class);
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo4 = Mockito.mock(SegmentLevelQuantizationInfo.class);

        try (
            MockedStatic<FieldInfoExtractor> fieldInfoExtractorMock = Mockito.mockStatic(FieldInfoExtractor.class);
            MockedStatic<SegmentLevelQuantizationUtil> segmentLevelQuantUtilMock = Mockito.mockStatic(SegmentLevelQuantizationUtil.class)
        ) {

            // Setup for non-empty quantization config
            QuantizationConfig nonEmptyConfig = QuantizationConfig.builder().build();

            fieldInfoExtractorMock.when(() -> FieldInfoExtractor.extractQuantizationConfig(fieldInfo4, testVersion))
                .thenReturn(nonEmptyConfig);

            // Setup for ADC disabled
            segmentLevelQuantUtilMock.when(() -> SegmentLevelQuantizationUtil.isAdcEnabled(segmentLevelQuantizationInfo4))
                .thenReturn(false);

            // Directly call the method
            VectorDataType result4 = knnIndexShard.determineVectorDataType(fieldInfo4, segmentLevelQuantizationInfo4, testVersion);

            // Verify BINARY is returned when ADC is disabled
            assertEquals(VectorDataType.BINARY, result4);
        }
    }
}
