/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

public class EngineResolverTests extends KNNTestCase {

    private static final EngineResolver ENGINE_RESOLVER = EngineResolver.INSTANCE;

    public void testResolveEngine_whenEngineSpecifiedInMethod_thenThatEngine() {
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                false
            )
        );
    }

    public void testResolveEngine_whenRequiresTraining_thenFaiss() {
        assertEquals(KNNEngine.FAISS, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, true));
    }

    public void testResolveEngine_whenModeAndCompressionAreFalse_thenDefault() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, false));
        assertEquals(
            KNNEngine.DEFAULT,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false
            )
        );
    }

    public void testResolveEngine_whenModeSpecifiedAndCompressionIsNotSpecified_whenVersionBefore2_19_thenNMSLIB() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, false));
        assertEquals(
            KNNEngine.NMSLIB,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false,
                Version.V_2_18_0
            )
        );
    }

    public void testResolveEngine_whenModeSpecifiedAndCompressionIsNotSpecified_thenFAISS() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, false));
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false
            )
        );
    }

    public void testResolveEngine_whenCompressionIs1x_thenEngineBasedOnMode() {
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x1).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build(), null, false)
        );
        assertEquals(
            KNNEngine.NMSLIB,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build(),
                null,
                false,
                Version.V_2_18_0
            )
        );
    }

    public void testResolveEngine_whenCompressionIs4x_thenEngineIsLucene() {
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x4).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x4).build(), null, false)
        );
    }

    public void testResolveEngine_whenConfiguredForBQ_thenEngineIsFaiss() {
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x2).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x2).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x8).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x8).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x16).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x16).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x32).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x32).build(),
                null,
                false
            )
        );
    }
}
