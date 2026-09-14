/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;

public class LuceneFlatMethodResolverTests extends KNNTestCase {

    private static final LuceneFlatMethodResolver TEST_RESOLVER = new LuceneFlatMethodResolver();

    public void testResolveMethod_whenFlatMethod_thenResolveWithX32Compression() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.L2, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x32, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenFlatMethodWithExplicitX32Compression_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .compressionLevel(CompressionLevel.x32)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.COSINESIMIL
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(CompressionLevel.x32, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenFlatMethodWithX16Compression_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .compressionLevel(CompressionLevel.x16)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.L2
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(CompressionLevel.x16, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenFlatMethodWithX8Compression_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .compressionLevel(CompressionLevel.x8)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(CompressionLevel.x8, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenFlatMethodWithUnsupportedCompression_thenThrow() {
        final Set<CompressionLevel> supported = Set.of(CompressionLevel.x8, CompressionLevel.x16, CompressionLevel.x32);
        for (CompressionLevel level : CompressionLevel.values()) {
            if (supported.contains(level) || level == CompressionLevel.NOT_CONFIGURED) {
                continue;
            }
            KNNMethodContext flatMethodContext = new KNNMethodContext(
                KNNEngine.LUCENE,
                SpaceType.L2,
                new MethodComponentContext(METHOD_FLAT, Map.of())
            );
            ValidationException ex = expectThrows(
                ValidationException.class,
                () -> TEST_RESOLVER.resolveMethod(
                    flatMethodContext,
                    KNNMethodConfigContext.builder()
                        .vectorDataType(VectorDataType.FLOAT)
                        .compressionLevel(level)
                        .versionCreated(Version.CURRENT)
                        .build(),
                    false,
                    SpaceType.L2
                )
            );
            final String msg = ex.getMessage();
            assertTrue("level=" + level + " msg=" + msg, msg.contains("[" + METHOD_FLAT + "]"));
            assertTrue("level=" + level + " msg=" + msg, msg.contains("only supports these compression levels"));
            // The message should list the supported levels — spot-check each name is present.
            for (CompressionLevel supportedLevel : supported) {
                assertTrue("level=" + level + " supportedLevel=" + supportedLevel + " msg=" + msg, msg.contains(supportedLevel.getName()));
            }
        }
    }

    public void testResolveMethod_whenFlatMethodWithParameters_thenThrow() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of("some_param", 10))
        );
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                flatMethodContext,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
                false,
                SpaceType.L2
            )
        );
    }

    public void testResolveMethod_whenFlatMethodWithMode_thenThrow() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                flatMethodContext,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                flatMethodContext,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.IN_MEMORY)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );
    }

    public void testResolveMethod_whenFlatMethodWithTraining_thenThrow() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                flatMethodContext,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
                true,
                SpaceType.L2
            )
        );
    }

    public void testResolveMethod_whenFlatMethodWithHalfFloat_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.HALF_FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.L2, resolvedMethodContext.getKnnMethodContext().getSpaceType());
    }

    /**
     * HALF_FLOAT must not inherit FLOAT's x32 default: absent an explicit choice, it defaults to x16
     * (SQ 1-bit on 16-bit storage) - unlike {@link LuceneHNSWMethodResolver}, which defaults HALF_FLOAT
     * to x1 (exact FP16, no SQ) instead.
     */
    public void testResolveMethod_whenFlatMethodWithHalfFloat_thenCompressionX16() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.HALF_FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );

        assertEquals(CompressionLevel.x16, resolvedMethodContext.getCompressionLevel());

        // FLOAT keeps the existing x32 default and its rescore behaviour.
        ResolvedMethodContext floatContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x32, floatContext.getCompressionLevel());
    }

    /**
     * HALF_FLOAT may opt into x1 explicitly: exact FP16 storage, no further reduction (the flat
     * method's default is x16, so this only applies when x1 is explicitly requested).
     */
    public void testResolveMethod_whenFlatMethodWithHalfFloatAndExplicitX1_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.HALF_FLOAT)
                .compressionLevel(CompressionLevel.x1)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenFlatMethodWithHalfFloatAndExplicitX16_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.HALF_FLOAT)
                .compressionLevel(CompressionLevel.x16)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x16, resolvedMethodContext.getCompressionLevel());
    }

    /** Every compression level besides x1 (opt-in exact FP16) and x16 (default, SQ 1-bit) must still be rejected for HALF_FLOAT. */
    public void testResolveMethod_whenFlatMethodWithHalfFloatAndUnsupportedCompression_thenThrow() {
        for (CompressionLevel level : CompressionLevel.values()) {
            if (level == CompressionLevel.NOT_CONFIGURED
                || LuceneFlatMethodResolver.SUPPORTED_COMPRESSION_LEVELS_HALF_FLOAT.contains(level)) {
                continue;
            }
            KNNMethodContext flatMethodContext = new KNNMethodContext(
                KNNEngine.LUCENE,
                SpaceType.L2,
                new MethodComponentContext(METHOD_FLAT, Map.of())
            );
            expectThrows(
                ValidationException.class,
                () -> TEST_RESOLVER.resolveMethod(
                    flatMethodContext,
                    KNNMethodConfigContext.builder()
                        .vectorDataType(VectorDataType.HALF_FLOAT)
                        .compressionLevel(level)
                        .versionCreated(Version.CURRENT)
                        .build(),
                    false,
                    SpaceType.L2
                )
            );
        }
    }

    public void testResolveMethod_whenFlatMethodWithHalfFloatAndInnerProduct_thenResolve() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.HALF_FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_FLAT, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
    }

    public void testResolveMethod_whenHalfFloatWithMode_thenThrows() {
        for (Mode mode : new Mode[] { Mode.ON_DISK, Mode.IN_MEMORY }) {
            expectThrows(
                ValidationException.class,
                () -> TEST_RESOLVER.resolveMethod(
                    new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, new MethodComponentContext(METHOD_FLAT, java.util.Map.of())),
                    KNNMethodConfigContext.builder()
                        .vectorDataType(VectorDataType.HALF_FLOAT)
                        .mode(mode)
                        .versionCreated(Version.CURRENT)
                        .build(),
                    false,
                    SpaceType.L2
                )
            );
        }
    }
}
