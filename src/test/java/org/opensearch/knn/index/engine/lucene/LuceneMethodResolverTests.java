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
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class LuceneMethodResolverTests extends KNNTestCase {
    MethodResolver TEST_RESOLVER = new LuceneMethodResolver();

    public void testResolveMethod_whenValid_thenResolve() {
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .mode(Mode.ON_DISK)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .compressionLevel(CompressionLevel.x4)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .mode(Mode.ON_DISK)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .compressionLevel(CompressionLevel.x4)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of())))
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.BYTE).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertFalse(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenInvalid_thenThrow() {
        // Invalid training context
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
                true,
                SpaceType.L2
            )
        );

        // Changed from 32x to 16x, Lucene 32x compression was added
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .compressionLevel(CompressionLevel.x16)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );

        // Invalid spec ondisk and compression is 1
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .compressionLevel(CompressionLevel.x1)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );
    }
}
