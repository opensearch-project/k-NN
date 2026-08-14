/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

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

        @Override
        public EncoderType getEncoderType() {
            return EncoderType.FLAT;
        }

        @Override
        public Set<QuantizationBits> getSupportedBits() {
            return EnumSet.of(QuantizationBits.FULL_PRECISION);
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

    public void testGetDefaultCompressionLevel_whenCompressionConfigured_thenReturnConfigured() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder()
            .compressionLevel(CompressionLevel.x16)
            .mode(Mode.ON_DISK)
            .versionCreated(Version.V_3_6_0)
            .build();
        assertEquals(CompressionLevel.x16, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenOnDiskAndV360OrLater_thenReturnX32() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.ON_DISK).versionCreated(Version.V_3_6_0).build();
        assertEquals(CompressionLevel.x32, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenOnDiskAndBeforeV360_thenReturnFallback() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.ON_DISK).versionCreated(Version.V_2_17_0).build();
        assertEquals(CompressionLevel.x4, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenOnDiskAndNullVersion_thenReturnFallback() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.ON_DISK).build();
        assertEquals(CompressionLevel.x4, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenNotOnDisk_thenReturnX1() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).versionCreated(Version.V_3_6_0).build();
        assertEquals(CompressionLevel.x1, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenNotConfigured_thenReturnX1() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().versionCreated(Version.V_3_6_0).build();
        assertEquals(CompressionLevel.x1, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV390OrLaterAndNoMode_thenReturnX32() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().versionCreated(Version.V_3_9_0).build();
        assertEquals(CompressionLevel.x32, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV390OrLaterAndInMemory_thenReturnX32() {
        // The flip overrides mode: an unspecified compression on V_3_9_0+ resolves to x32 even for IN_MEMORY.
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).versionCreated(Version.V_3_9_0).build();
        assertEquals(CompressionLevel.x32, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV390OrLaterAndExplicitX1_thenReturnX1() {
        // An explicit compression_level is always honored, even after the flip.
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder()
            .compressionLevel(CompressionLevel.x1)
            .versionCreated(Version.V_3_9_0)
            .build();
        assertEquals(CompressionLevel.x1, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV380AndNoMode_thenReturnX1() {
        // Before the flip version, an unspecified compression with no on-disk mode stays x1.
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().versionCreated(Version.V_3_8_0).build();
        assertEquals(CompressionLevel.x1, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV380AndOnDisk_thenReturnX32() {
        // Existing ON_DISK behavior for versions between V_3_6_0 and the flip is unchanged.
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.ON_DISK).versionCreated(Version.V_3_8_0).build();
        assertEquals(CompressionLevel.x32, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testGetDefaultCompressionLevel_whenV350AndOnDisk_thenReturnFallback() {
        KNNMethodConfigContext ctx = KNNMethodConfigContext.builder().mode(Mode.ON_DISK).versionCreated(Version.V_3_5_0).build();
        assertEquals(CompressionLevel.x4, TEST_RESOLVER.getDefaultCompressionLevel(ctx, CompressionLevel.x4));
    }

    public void testShouldEncoderBeResolved_whenV390OrLaterAndDefaultInMemory_thenResolve() {
        // Before the flip, an unspecified compression + non-ON_DISK mode short-circuits encoder resolution.
        // On V_3_9_0+, the encoder must be resolved so the x32 default takes effect.
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.V_3_8_0).build()
            )
        );
        assertTrue(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.V_3_9_0).build()
            )
        );
        assertTrue(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.IN_MEMORY)
                    .versionCreated(Version.V_3_9_0)
                    .build()
            )
        );
        // An explicit x1 still short-circuits, even on V_3_9_0+.
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .compressionLevel(CompressionLevel.x1)
                    .versionCreated(Version.V_3_9_0)
                    .build()
            )
        );
        // Non-FLOAT vectors are never auto-encoded, even on V_3_9_0+.
        assertFalse(
            TEST_RESOLVER.shouldEncoderBeResolved(
                null,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.BINARY).versionCreated(Version.V_3_9_0).build()
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
