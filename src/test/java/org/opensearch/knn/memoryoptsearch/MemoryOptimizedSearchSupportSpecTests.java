/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class MemoryOptimizedSearchSupportSpecTests extends KNNTestCase {
    private static final Optional<String> NO_MODEL_ID = Optional.empty();
    private static final QuantizationConfig NO_QUANTIZATION = null;

    public void testLuceneEngineIsSupported() {
        // Lucene + any configurations must be supported.
        mustSupported(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.INNER_PRODUCT,
                VectorDataType.FLOAT,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustSupported(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.INNER_PRODUCT,
                VectorDataType.BYTE,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustSupported(
            new TestingSpec(
                KNNEngine.LUCENE,
                SpaceType.L2,
                VectorDataType.FLOAT,
                mock(MethodComponentContext.class),
                NO_QUANTIZATION,
                NO_MODEL_ID
            )
        );
        mustSupported(
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

    public void testFaissSupportedCases() {
        // HNSW,float, L2|IP, Flat
        // HNSW,float, L2|IP, SQ
        // HNSW,binary, Hamming, binary
        // Note that we do support byte index. And it is VectorDataType.FLOAT for the byte index, not VectorDataType.BYTE.

        mustSupported(
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

        mustSupported(
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

        mustSupported(
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

        mustSupported(
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

        mustSupported(
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
        mustSupported(
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

        mustSupported(
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

        mustSupported(
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
        mustNotSupported(
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
        mustNotSupported(
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

    public void testPQNotSupported() {
        // Non-empty model id
        mustNotSupported(
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

    private void mustSupported(final TestingSpec testingSpec) {
        doTest(testingSpec, true);
    }

    private void mustNotSupported(final TestingSpec testingSpec) {
        doTest(testingSpec, false);
    }

    private void doTest(final TestingSpec testingSpec, final boolean expected) {
        final boolean isSupported = MemoryOptimizedSearchSupportSpec.supported(
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
