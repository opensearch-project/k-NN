/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.Builder;
import lombok.RequiredArgsConstructor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BINARY;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class MemoryOptimizedSearchSupportSpecTests extends KNNTestCase {
    private static final Optional<String> NO_MODEL_ID = Optional.empty();

    public void testLuceneEngineIsSupported() {
        // Lucene + any configurations must be supported.
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.LUCENE,
            Arrays.asList(SpaceType.values()),
            Arrays.asList(VectorDataType.values()),
            // Don't care MethodComponentContext for Lucene
            Collections.emptyList(),
            NO_MODEL_ID
        );

        mustSupported(testingSpec);
    }

    public void testFaissSupportedCases() {
        // HNSW,float, L2|IP, Flat
        // HNSW,float, L2|IP, SQ
        // Note that we do support byte index. And it is VectorDataType.FLOAT for the byte index, not VectorDataType.BYTE.
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.FAISS,
            Arrays.asList(SpaceType.L2, SpaceType.INNER_PRODUCT),
            Arrays.asList(VectorDataType.FLOAT),
            Arrays.asList(
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                ),
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                )
            ),
            NO_MODEL_ID
        );

        mustSupported(testingSpec);
    }

    public void testFaissUnsupportedCases() {
        // HNSW,float, L2|IP, Flat
        // HNSW,float, L2|IP, SQ
        // Note that we do support byte index. And it is VectorDataType.FLOAT for the byte index, not VectorDataType.BYTE.
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.FAISS,
            Arrays.asList(SpaceType.values()),
            Arrays.asList(VectorDataType.values()),
            Arrays.asList(
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext("DUMMY_KEY", Collections.emptyMap()))
                ),
                new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, new Object()))
            ),
            NO_MODEL_ID
        );

        mustNotSupported(testingSpec);
    }

    public void testBinaryFiassNotSupportedCases() {
        // HNSW,binary, L2|IP, Flat
        // HNSW,binary, L2|IP, SQ
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.FAISS,
            Arrays.asList(SpaceType.values()),
            Arrays.asList(VectorDataType.BINARY),
            Arrays.asList(
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext("DUMMY_KEY", Collections.emptyMap()))
                ),
                new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, new Object()))
            ),
            NO_MODEL_ID
        );

        mustNotSupported(testingSpec);
    }

    public void testBinaryFiassSupportedCases() {
        // HNSW,binary, L2|IP, Flat
        // HNSW,binary, L2|IP, SQ (=disk mode 2x)
        // HNSW,binary, L2|IP, binary
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.FAISS,
            Arrays.asList(SpaceType.values()),
            Arrays.asList(VectorDataType.BINARY),
            Arrays.asList(
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                ),
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_BINARY, Collections.emptyMap()))
                ),
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Collections.emptyMap()))
                )
            ),
            NO_MODEL_ID
        );

        mustSupported(testingSpec);
    }

    public void testPQNotSupported() {
        final TestingSpec testingSpec = new TestingSpec(
            KNNEngine.FAISS,
            Arrays.asList(SpaceType.values()),
            Arrays.asList(VectorDataType.FLOAT),
            Arrays.asList(
                new MethodComponentContext(
                    METHOD_HNSW,
                    Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_FLAT, Collections.emptyMap()))
                )
            ),
            Optional.of("model_id")
        );

        mustNotSupported(testingSpec);
    }

    private void mustSupported(final TestingSpec testingSpec) {
        doTest(testingSpec, true);
    }

    private void mustNotSupported(final TestingSpec testingSpec) {
        doTest(testingSpec, false);
    }

    private void doTest(final TestingSpec testingSpec, final boolean expected) {
        for (final SpaceType spaceType : testingSpec.spaceTypes) {
            for (final VectorDataType vectorDataType : testingSpec.vectorDataTypes) {
                for (final MethodComponentContext methodComponentContext : testingSpec.methodComponentContexts) {
                    final Params params = buildParameters(testingSpec.knnEngine, spaceType, vectorDataType, methodComponentContext);

                    final boolean isSupported = MemoryOptimizedSearchSupportSpec.supported(params.methodContextOpt, testingSpec.modelId);

                    assertEquals(expected, isSupported);
                }
            }
        }
    }

    private Params buildParameters(
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final VectorDataType vectorDataType,
        final MethodComponentContext methodComponentContext
    ) {

        final Params.ParamsBuilder builder = Params.builder();
        builder.vectorDataType(vectorDataType);

        final KNNMethodContext methodContext = new KNNMethodContext(knnEngine, spaceType, methodComponentContext);
        builder.methodContextOpt = Optional.of(methodContext);

        return builder.build();
    }

    @Builder
    private static class Params {
        Optional<KNNMethodContext> methodContextOpt;
        QuantizationConfig quantizationConfig;
        VectorDataType vectorDataType;
    }

    @RequiredArgsConstructor
    private static class TestingSpec {
        final KNNEngine knnEngine;
        final List<SpaceType> spaceTypes;
        final List<VectorDataType> vectorDataTypes;
        final List<MethodComponentContext> methodComponentContexts;
        final Optional<String> modelId;
    }
}
