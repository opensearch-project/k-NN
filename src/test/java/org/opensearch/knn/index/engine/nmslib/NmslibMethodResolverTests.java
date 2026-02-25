/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.nmslib;

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

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Deprecated(since = "2.19.0", forRemoval = true)
public class NmslibMethodResolverTests extends KNNTestCase {

    MethodResolver TEST_RESOLVER = new NmslibMethodResolver();

    public void testResolveMethod_whenValid_thenResolve() {
        // No configuration passed in
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertEquals(KNNEngine.NMSLIB, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.NMSLIB,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertEquals(KNNEngine.NMSLIB, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());
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

        // Invalid compression
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .compressionLevel(CompressionLevel.x8)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );

        // Invalid mode
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );
    }
}
