/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class AbstractMethodResolverTests extends KNNTestCase {

    private final static String ENCODER_NAME = "test";
    private final static CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.x8;

    private final static AbstractMethodResolver TEST_RESOLVER = new AbstractMethodResolver() {
        @Override
        public ResolvedMethodContext resolveMethod(
            KNNMethodContext knnMethodContext,
            KNNMethodConfigContext knnMethodConfigContext,
            boolean shouldRequireTraining,
            SpaceType spaceType
        ) {
            return null;
        }
    };

    private final static Encoder TEST_ENCODER = new Encoder() {

        @Override
        public MethodComponent getMethodComponent() {
            return MethodComponent.Builder.builder(ENCODER_NAME).build();
        }

        @Override
        public CompressionLevel calculateCompressionLevel(
            MethodComponentContext encoderContext,
            KNNMethodConfigContext knnMethodConfigContext
        ) {
            return DEFAULT_COMPRESSION;
        }
    };

    private final static Map<String, Encoder> ENCODER_MAP = Map.of(ENCODER_NAME, TEST_ENCODER);

    public void testResolveCompressionLevelFromMethodContext() {
        assertEquals(
            CompressionLevel.x1,
            TEST_RESOLVER.resolveCompressionLevelFromMethodContext(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                KNNMethodConfigContext.builder().build(),
                ENCODER_MAP
            )
        );
        assertEquals(
            DEFAULT_COMPRESSION,
            TEST_RESOLVER.resolveCompressionLevelFromMethodContext(
                new KNNMethodContext(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    new MethodComponentContext(
                        METHOD_HNSW,
                        Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_NAME, Map.of()))
                    )
                ),
                KNNMethodConfigContext.builder().build(),
                ENCODER_MAP
            )
        );
    }

    public void testIsEncoderSpecified() {
        assertFalse(TEST_RESOLVER.isEncoderSpecified(null));
        assertFalse(
            TEST_RESOLVER.isEncoderSpecified(new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY))
        );
        assertFalse(
            TEST_RESOLVER.isEncoderSpecified(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, new MethodComponentContext(METHOD_HNSW, Map.of()))
            )
        );
        assertTrue(
            TEST_RESOLVER.isEncoderSpecified(
                new KNNMethodContext(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, "test"))
                )
            )
        );
    }

    public void testShouldEncoderBeResolved() {
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                new KNNMethodContext(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, "test"))
                ),
                KNNMethodConfigContext.builder().build()
            )
        );
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(null, KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build())
        );
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).mode(Mode.ON_DISK).build()
            )
        );
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.NOT_CONFIGURED).mode(Mode.IN_MEMORY).build()
            )
        );
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder()
                    .compressionLevel(CompressionLevel.NOT_CONFIGURED)
                    .mode(Mode.ON_DISK)
                    .vectorDataType(VectorDataType.BINARY)
                    .build()
            )
        );
        assertTrue(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder()
                    .compressionLevel(CompressionLevel.NOT_CONFIGURED)
                    .mode(Mode.ON_DISK)
                    .vectorDataType(VectorDataType.FLOAT)
                    .build()
            )
        );
        assertTrue(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder()
                    .compressionLevel(CompressionLevel.x32)
                    .mode(Mode.ON_DISK)
                    .vectorDataType(VectorDataType.FLOAT)
                    .build()
            )
        );
    }
}
