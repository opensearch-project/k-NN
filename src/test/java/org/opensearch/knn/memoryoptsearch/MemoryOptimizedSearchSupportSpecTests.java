/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class MemoryOptimizedSearchSupportSpecTests extends KNNTestCase {
    private static final Optional<String> NO_MODEL_ID = Optional.empty();
    private static final QuantizationConfig NO_QUANTIZATION = null;

    public void testLuceneEngineIsIsSupportedFieldType() {
        // Lucene + any configurations must be supported.
        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.INNER_PRODUCT,
                VectorDataType.FLOAT,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.INNER_PRODUCT,
                VectorDataType.BYTE,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.L2,
                VectorDataType.FLOAT,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.L2,
                VectorDataType.BYTE,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
    }

    public void testFaissIsSupportedFieldTypeCases() {
        // HNSW,float, L2|IP, Flat
        // HNSW,float, L2|IP, SQ
        // HNSW,binary, Hamming, binary
        // Note that we do support byte index. And it is VectorDataType.FLOAT for the byte index, not VectorDataType.BYTE.

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_BINARY, Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
    }

    public void testFaissQuantizationCases() {
        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                ),
                QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build(),
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                ),
                QuantizationConfig.builder().quantizationType(ScalarQuantizationType.TWO_BIT).build(),
                NO_MODEL_ID
            )
        );

        mustIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.HAMMING,
                VectorDataType.BINARY,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                ),
                QuantizationConfig.builder().quantizationType(ScalarQuantizationType.FOUR_BIT).build(),
                NO_MODEL_ID
            )
        );
    }

    public void testFaissUnsupportedCases() {
        // Unsupported encoding
        mustNotIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext("DUMMY_KEY", Collections.emptyMap()))
                ),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );

        // Invalid encoder type
        mustNotIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, new Object())),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
    }

    public void testPQNotIsSupportedFieldType() {
        // Non-empty model id
        mustNotIsSupportedFieldType(
            new TestingSpec(
                KNNEngine.FAISS,
                SpaceType.INNER_PRODUCT,
                VectorDataType.FLOAT,
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                ),
                mock(QuantizationConfig.class),
                Optional.of("model_id")
            )
        );
    }

    public void testIsSupportedFieldTypeDuringSearch() {
        // |----------------------|-------------|---------------||-----------|
        // | field type supported | mem_opt_src | on_disk && 1x || supported |
        // |----------------------|-------------|---------------||-----------|
        // | true | true | true || true |
        // | true | true | false || true |
        // | true | false | true || true |
        // | true | false | false || false |
        // | false | true | true || false |
        // | false | true | false || false |
        // | false | false | true || false |
        // | false | false | false || false |
        // |----------------------|-------------|---------------||-----------|

        doTestIsSupportedFieldTypeDuringSearch(true, true, true, true);
        doTestIsSupportedFieldTypeDuringSearch(true, true, false, true);
        doTestIsSupportedFieldTypeDuringSearch(true, false, true, true);
        doTestIsSupportedFieldTypeDuringSearch(true, false, false, false);
        doTestIsSupportedFieldTypeDuringSearch(false, true, true, false);
        doTestIsSupportedFieldTypeDuringSearch(false, true, false, false);
        doTestIsSupportedFieldTypeDuringSearch(false, false, true, false);
        doTestIsSupportedFieldTypeDuringSearch(false, false, false, false);
    }

    public void doTestIsSupportedFieldTypeDuringSearch(
        final boolean fieldTypeSupported,
        final boolean memoryOptSrchSupported,
        final boolean onDiskWith1x,
        final boolean expected
    ) {
        try (MockedStatic<KNNSettings> knnSettingsMockedStatic = mockStatic(KNNSettings.class)) {
            knnSettingsMockedStatic.when(() -> KNNSettings.isMemoryOptimizedKnnSearchModeEnabled(any())).thenReturn(memoryOptSrchSupported);

            final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
            when(fieldType.isMemoryOptimizedSearchAvailable()).thenReturn(fieldTypeSupported);

            final KNNMappingConfig mappingConfig = mock(KNNMappingConfig.class);
            if (onDiskWith1x) {
                when(mappingConfig.getMode()).thenReturn(Mode.ON_DISK);
                when(mappingConfig.getCompressionLevel()).thenReturn(CompressionLevel.x1);
            } else {
                when(mappingConfig.getMode()).thenReturn(Mode.NOT_CONFIGURED);
                when(mappingConfig.getCompressionLevel()).thenReturn(CompressionLevel.NOT_CONFIGURED);
            }
            when(fieldType.getKnnMappingConfig()).thenReturn(mappingConfig);

            assertEquals(expected, MemoryOptimizedSearchSupportSpec.isSupportedFieldType(fieldType, "IndexName"));
        }
    }

    private void mustIsSupportedFieldType(final TestingSpec testingSpec) {
        doTest(testingSpec, true);
    }

    private void mustNotIsSupportedFieldType(final TestingSpec testingSpec) {
        doTest(testingSpec, false);
    }

    private void doTest(final TestingSpec testingSpec, final boolean expected) {
        final boolean isSupported = MemoryOptimizedSearchSupportSpec.isSupportedFieldType(
            testingSpec.methodComponentContext,
            testingSpec.quantizationConfig,
            testingSpec.modelId
        );
        assertEquals(expected, isSupported);
    }

    private static class TestingSpec {
        final KNNEngine knnEngine;
        final SpaceType spaceType;
        final VectorDataType vectorDataType;
        final Optional<KNNMethodContext> methodComponentContext;
        final QuantizationConfig quantizationConfig;
        final Optional<String> modelId;

        private TestingSpec(
            final KNNEngine knnEngine,
            final SpaceType spaceType,
            final VectorDataType vectorDataType,
            final MethodComponentContext methodComponentContext,
            final QuantizationConfig quantizationConfig,
            final Optional<String> modelId
        ) {
            this.knnEngine = knnEngine;
            this.spaceType = spaceType;
            this.vectorDataType = vectorDataType;
            final KNNMethodContext methodContext = new KNNMethodContext(knnEngine, spaceType, methodComponentContext);
            this.methodComponentContext = Optional.of(methodContext);
            this.quantizationConfig = quantizationConfig;
            this.modelId = modelId;
        }
    }
}
