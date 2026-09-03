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
}
