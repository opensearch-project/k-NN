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

    public void testResolveMethod_whenFlatMethodWithUnsupportedCompression_thenThrow() {
        for (CompressionLevel level : CompressionLevel.values()) {
            if (level == CompressionLevel.x32 || level == CompressionLevel.NOT_CONFIGURED) {
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
                        .vectorDataType(VectorDataType.FLOAT)
                        .compressionLevel(level)
                        .versionCreated(Version.CURRENT)
                        .build(),
                    false,
                    SpaceType.L2
                )
            );
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
     * HALF_FLOAT must not inherit the x32 default: it does not go through an encoder, and x32
     * combined with the flat method switches on a default rescore that cannot change an exhaustive
     * FP16 ranking.
     */
    public void testResolveMethod_whenFlatMethodWithHalfFloat_thenCompressionX1AndNoRescore() {
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

        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
        assertNull(
            "half_float flat must not default to a rescore pass",
            resolvedMethodContext.getCompressionLevel().getDefaultRescoreContext()
        );

        // FLOAT keeps the existing x32 default and its rescore behaviour.
        ResolvedMethodContext floatContext = TEST_RESOLVER.resolveMethod(
            flatMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x32, floatContext.getCompressionLevel());
    }

    /** The default (x1) must also be accepted when set explicitly, not just by default. */
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

    /**
     * Every compression level outside {@link LuceneFlatMethodResolver#SUPPORTED_COMPRESSION_HALF_FLOAT}
     * must throw. Driving the loop off that set rather than a hardcoded level keeps this honest as
     * levels are added: a newly supported level drops out of the loop, the rest still have to fail.
     */
    public void testResolveMethod_whenFlatMethodWithHalfFloatAndUnsupportedCompression_thenThrow() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Map.of())
        );
        for (CompressionLevel level : CompressionLevel.values()) {
            if (level == CompressionLevel.NOT_CONFIGURED || LuceneFlatMethodResolver.SUPPORTED_COMPRESSION_HALF_FLOAT.contains(level)) {
                continue;
            }
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
}
