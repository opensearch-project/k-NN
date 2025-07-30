/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class EngineResolverTests extends KNNTestCase {

    private static final EngineResolver ENGINE_RESOLVER = EngineResolver.INSTANCE;

    public void testResolveEngine_whenEngineSpecifiedInMethod_thenThatEngine() {
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                null,
                false
            )
        );
    }

    public void testResolveEngine_whenRequiresTraining_thenFaiss() {
        assertEquals(KNNEngine.FAISS, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, null, true));
    }

    public void testResolveEngine_whenModeAndCompressionAreFalse_thenDefault() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, null, false));
        assertEquals(
            KNNEngine.DEFAULT,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                null,
                false
            )
        );
    }

    public void testResolveEngine_whenModeSpecifiedAndCompressionIsNotSpecified_whenVersionBefore2_19_thenNMSLIB() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, null, false));
        assertEquals(
            KNNEngine.NMSLIB,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                null,
                false,
                Version.V_2_18_0
            )
        );
    }

    public void testResolveEngine_whenModeSpecifiedAndCompressionIsNotSpecified_thenFAISS() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, null, false));
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                null,
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
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build(), null, null, false)
        );
        assertEquals(
            KNNEngine.NMSLIB,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build(),
                null,
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
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x4).build(), null, null, false)
        );
    }

    public void testResolveEngine_whenConfiguredForBQ_thenEngineIsFaiss() {
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x2).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x2).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x8).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x8).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x16).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x16).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x32).build(),
                null,
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x32).build(),
                null,
                null,
                false
            )
        );
    }

    public void testResolveEngine_whenMethodAndTopLevelEngineSpecified() throws IOException {
        // only method defined; set to faiss (default)
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.FAISS, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                null,
                false,
                Version.CURRENT
            )
        );

        // only method defined; set to lucene (non-default)
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                null,
                false,
                Version.CURRENT
            )
        );

        // method set to lucene, top level set to faiss; should throw exception
        expectThrows(
            MapperParsingException.class,
            () -> ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                "faiss",
                false,
                Version.CURRENT
            )
        );

        // method set to faiss, top level set to lucene; should throw exception
        expectThrows(
            MapperParsingException.class,
            () -> ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.FAISS, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                "lucene",
                false,
                Version.CURRENT
            )
        );

        // only top-level defined; set to faiss (default)
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, "faiss", false, Version.CURRENT)
        );

        // only top-level defined; set to lucene (non-default)
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, "lucene", false, Version.CURRENT)
        );

        // no engine defined; method not defined
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, "", false, Version.CURRENT)
        );

        // no engine defined; method defined
        String methodName = "test-method";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(PARAMETERS, (String) null)
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        assertEquals(
            KNNEngine.FAISS,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.UNDEFINED, SpaceType.DEFAULT, MethodComponentContext.EMPTY, false),
                "",
                false,
                Version.CURRENT
            )
        );
    }

    public void testValidateTopLevelEngine() throws IOException {
        // only top-level defined; set to faiss with compression 4x
        expectThrows(
            MapperParsingException.class,
            () -> ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x4).build(),
                null,
                "faiss",
                false,
                Version.CURRENT
            )
        );

        // top-level and method defined to lucene; requires training
        expectThrows(
            MapperParsingException.class,
            () -> ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x4).build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                "lucene",
                true,
                Version.CURRENT
            )
        );
    }
}
