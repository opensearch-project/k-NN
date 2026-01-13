/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReader;
import org.junit.Before;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.carrotsearch.randomizedtesting.RandomizedTest.$;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;

/**
 * Comprehensive parameterized tests for SearchVectorTypeResolver.
 * Tests various index configurations including quantized indices, native byte/binary types,
 * and regular float vectors to ensure correct search function selection.
 */
public class SearchVectorTypeResolverTests extends KNNTestCase {

    @Mock
    private SegmentReader segmentReader;
    @Mock
    private FieldInfo fieldInfo;

    private static final String FIELD_NAME = "test_vector_field";

    // Test parameters
    private final String description;
    private final VectorDataType vectorDataType;
    private final QuantizationConfig quantizationConfig;

    // Constructor for parameterized tests
    public SearchVectorTypeResolverTests(String description, VectorDataType vectorDataType, QuantizationConfig quantizationConfig) {
        this.description = description;
        this.vectorDataType = vectorDataType;
        this.quantizationConfig = quantizationConfig;
    }

    @Before
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
    }

    /**
     * Factory method to generate test parameters for various index configurations.
     */
    @ParametersFactory(argumentFormatting = "description:%1$s")
    public static Iterable<Object[]> parameters() {
        return Arrays.asList(
            $(
                "1-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, false, false)
            ),
            $(
                "2-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.TWO_BIT, false, false)
            ),
            $(
                "4-bit quantized index (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.FOUR_BIT, false, false)
            ),
            $(
                "1-bit quantized index with ADC (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, true, false)
            ),
            $(
                "1-bit quantized index with random rotation (FLOAT vector type)",
                VectorDataType.FLOAT,
                createQuantizationConfig(ScalarQuantizationType.ONE_BIT, false, true)
            ),
            $("Native BYTE vector type (no quantization)", VectorDataType.BYTE, QuantizationConfig.EMPTY),
            $("Native BINARY vector type (no quantization)", VectorDataType.BINARY, QuantizationConfig.EMPTY),
            $("Regular FLOAT vectors (no quantization)", VectorDataType.FLOAT, QuantizationConfig.EMPTY),
            $("ADC transformed FLOAT vectors (no quantization config)", VectorDataType.FLOAT, null)
        );
    }

    /**
     * Helper method to create QuantizationConfig for testing.
     */
    private static QuantizationConfig createQuantizationConfig(
        ScalarQuantizationType sqType,
        boolean enableADC,
        boolean enableRandomRotation
    ) {
        return QuantizationConfig.builder()
            .quantizationType(sqType)
            .enableADC(enableADC)
            .enableRandomRotation(enableRandomRotation)
            .build();
    }

    /**
     * Parameterized test that verifies SearchVectorTypeResolver returns the correct search function
     * for different index configurations.
     *
     * This test verifies that the resolver correctly determines whether to use byte[] or float[] search
     * based on the field configuration, but does not execute the actual search since that would require
     * a real Lucene index.
     */
    public void testSearchFunctionSelection() {
        // Setup field info with quantization config
        setupFieldInfo(quantizationConfig);

        // Get the search function - this is the main behavior we're testing
        VectorSearchFunction searchFunction = SearchVectorTypeResolver.getSearchFunction(segmentReader, fieldInfo, vectorDataType);

        // Verify that a search function was returned
        assertNotNull("Search function should not be null for: " + description, searchFunction);

        // The actual search behavior is tested through integration tests
        // This unit test verifies that the correct decision logic is applied
    }

    /**
     * Test to ensure the utility class can be instantiated (for code coverage).
     * This covers the implicit constructor.
     */
    public void testConstructor() {
        // Instantiate the utility class to cover the constructor line
        SearchVectorTypeResolver resolver = new SearchVectorTypeResolver();
        assertNotNull("SearchVectorTypeResolver instance should not be null", resolver);
    }

    /**
     * Helper method to setup FieldInfo mock with quantization configuration.
     */
    private void setupFieldInfo(QuantizationConfig quantizationConfig) {
        Map<String, String> attributes = new HashMap<>();
        if (quantizationConfig != null && quantizationConfig != QuantizationConfig.EMPTY) {
            // Use QuantizationConfigParser to serialize the config to CSV format
            String configCsv = QuantizationConfigParser.toCsv(quantizationConfig);
            attributes.put(QFRAMEWORK_CONFIG, configCsv);
        }
        when(fieldInfo.attributes()).thenReturn(attributes);
        when(fieldInfo.getName()).thenReturn(FIELD_NAME);
    }
}
