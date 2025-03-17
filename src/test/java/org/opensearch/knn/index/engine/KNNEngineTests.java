/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.faiss.FaissHNSWMethod;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;
import org.opensearch.remoteindexbuild.constants.KNNRemoteConstants;
import org.opensearch.remoteindexbuild.model.RemoteFaissHNSWIndexParameters;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.COMPOUND_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.index.SpaceType.L2;
import static org.opensearch.remoteindexbuild.constants.KNNRemoteConstants.METHOD_PARAMETER_ENCODER;

public class KNNEngineTests extends KNNTestCase {
    /**
     * Check that version from engine and library match
     */
    public void testDelegateLibraryFunctions() {
        assertEquals(Nmslib.INSTANCE.getVersion(), KNNEngine.NMSLIB.getVersion());
        assertEquals(Faiss.INSTANCE.getVersion(), KNNEngine.FAISS.getVersion());
        assertEquals(Lucene.INSTANCE.getVersion(), KNNEngine.LUCENE.getVersion());
    }

    public void testGetDefaultEngine_thenReturnFAISS() {
        assertEquals(KNNEngine.FAISS, KNNEngine.DEFAULT);
    }

    /**
     * Test name getter
     */
    public void testGetName() {
        assertEquals(NMSLIB_NAME, KNNEngine.NMSLIB.getName());
    }

    /**
     * Test engine getter
     */
    public void testGetEngine() {
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngine(NMSLIB_NAME));
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngine("invalid"));
    }

    public void testGetEngineFromPath() {
        String hnswPath1 = "test" + Nmslib.EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath1));
        String hnswPath2 = "test" + Nmslib.EXTENSION + COMPOUND_EXTENSION;
        assertEquals(KNNEngine.NMSLIB, KNNEngine.getEngineNameFromPath(hnswPath2));

        String faissPath1 = "test" + FAISS_EXTENSION;
        assertEquals(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath(faissPath1));
        String faissPath2 = "test" + FAISS_EXTENSION + COMPOUND_EXTENSION;
        assertEquals(KNNEngine.FAISS, KNNEngine.getEngineNameFromPath(faissPath2));

        String invalidPath = "test.invalid";
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngineNameFromPath(invalidPath));
    }

    public void testMmapFileExtensions() {
        final List<String> mmapExtensions = Arrays.stream(KNNEngine.values())
            .flatMap(engine -> engine.mmapFileExtensions().stream())
            .collect(Collectors.toList());
        assertNotNull(mmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(mmapExtensions));
        assertTrue(mmapExtensions.containsAll(expectedSettings));
    }

    /**
     * The remote build service currently only supports HNSWFlat.
     */
    public void testSupportsRemoteIndexBuild() {
        KNNEngine Faiss = KNNEngine.FAISS;
        KNNEngine Lucene = KNNEngine.LUCENE;

        KNNMethodContext faissHNSWFlat = createMockMethodContext();
        KNNMethodContext faissIVFFlat = createFaissIVFMethodContext();
        KNNMethodContext luceneHNSWFlat = createLuceneHNSWMethodContext();

        assertTrue(Faiss.supportsRemoteIndexBuild(faissHNSWFlat.getMethodComponentContext()));
        assertFalse(Faiss.supportsRemoteIndexBuild(faissIVFFlat.getMethodComponentContext()));
        assertFalse(Lucene.supportsRemoteIndexBuild(luceneHNSWFlat.getMethodComponentContext()));
    }

    public void testCreateRemoteIndexingParameters_Success() {
        RemoteIndexParameters result = FaissHNSWMethod.createRemoteIndexingParameters(createMockMethodContext());

        assertNotNull(result);
        assertTrue(result instanceof RemoteFaissHNSWIndexParameters);

        RemoteFaissHNSWIndexParameters hnswParams = (RemoteFaissHNSWIndexParameters) result;

        assertEquals(METHOD_HNSW, hnswParams.getAlgorithm());
        assertEquals(L2.getValue(), hnswParams.getSpaceType());
        assertEquals(94, hnswParams.getEfConstruction());
        assertEquals(89, hnswParams.getEfSearch());
        assertEquals(14, hnswParams.getM());
    }

    public static KNNMethodContext createFaissIVFMethodContext() {
        MethodComponentContext encoder = new MethodComponentContext(ENCODER_SQ, Map.of());
        Map<String, Object> encoderMap = Map.of(METHOD_ENCODER_PARAMETER, encoder);
        Map<String, Object> parameters = Map.of(
            METHOD_PARAMETER_EF_SEARCH,
            24,
            METHOD_PARAMETER_EF_CONSTRUCTION,
            28,
            METHOD_PARAMETER_M,
            12,
            METHOD_ENCODER_PARAMETER,
            encoderMap
        );
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_IVF, parameters);
        return new KNNMethodContext(KNNEngine.FAISS, L2, methodComponentContext);
    }

    public static KNNMethodContext createLuceneHNSWMethodContext() {
        Map<String, Object> parameters = Map.of(METHOD_PARAMETER_EF_CONSTRUCTION, 28, METHOD_PARAMETER_M, 12);
        MethodComponentContext methodComponentContext = new MethodComponentContext(METHOD_HNSW, parameters);
        return new KNNMethodContext(KNNEngine.LUCENE, L2, methodComponentContext);
    }

    public static KNNMethodContext createMockMethodContext() {
        MethodComponentContext encoder = new MethodComponentContext(KNNConstants.ENCODER_FLAT, Map.of());
        Map<String, Object> parameters = Map.of(
            KNNRemoteConstants.METHOD_PARAMETER_EF_SEARCH,
            89,
            KNNRemoteConstants.METHOD_PARAMETER_EF_CONSTRUCTION,
            94,
            KNNRemoteConstants.METHOD_PARAMETER_M,
            14,
            METHOD_PARAMETER_ENCODER,
            encoder
        );
        MethodComponentContext methodComponentContext = new MethodComponentContext(KNNConstants.METHOD_HNSW, parameters);
        return new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, methodComponentContext);
    }

}
