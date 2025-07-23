/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.SneakyThrows;
import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.faiss.Faiss;
import org.opensearch.knn.index.engine.faiss.FaissHNSWMethod;
import org.opensearch.knn.index.engine.lucene.Lucene;
import org.opensearch.knn.index.engine.nmslib.Nmslib;
import org.opensearch.remoteindexbuild.model.RemoteFaissHNSWIndexParameters;
import org.opensearch.remoteindexbuild.model.RemoteIndexParameters;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.COMPOUND_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_EXTENSION;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.NMSLIB_NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.SpaceType.COSINESIMIL;
import static org.opensearch.knn.index.SpaceType.INNER_PRODUCT;
import static org.opensearch.knn.index.SpaceType.L2;

public class KNNEngineTests extends KNNTestCase {
    /**
     * Check that version from engine and library match
     */
    public void testDelegateLibraryFunctions() {
        assertEquals(Nmslib.INSTANCE.getVersion(), KNNEngine.NMSLIB.getVersion());
        assertEquals(Faiss.INSTANCE.getVersion(), KNNEngine.FAISS.getVersion());
        assertEquals(Lucene.INSTANCE.getVersion(), KNNEngine.LUCENE.getVersion());

        // Validate that deprecated engines have correct deprecation versions
        assertTrue(KNNEngine.NMSLIB.getRestrictedFromVersion() != null);
        assertFalse(KNNEngine.FAISS.isRestricted(Version.V_3_0_0)); // FAISS should not be deprecated
    }

    /**
     * Test that deprecated engines are correctly flagged
     */
    public void testIsRestricted() {
        Version deprecatedVersion = KNNEngine.NMSLIB.getRestrictedFromVersion();
        assertNotNull(deprecatedVersion);
        assertTrue(KNNEngine.NMSLIB.isRestricted(Version.V_3_0_0)); // Should return true for later versions

        assertFalse(KNNEngine.FAISS.isRestricted(Version.V_2_19_0)); // FAISS should not be deprecated
        assertFalse(KNNEngine.LUCENE.isRestricted(Version.V_2_19_0)); // LUCENE should not be deprecated
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
            .filter(engine -> engine != KNNEngine.UNDEFINED)
            .flatMap(engine -> engine.mmapFileExtensions().stream())
            .collect(Collectors.toList());
        assertNotNull(mmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(mmapExtensions));
        assertTrue(mmapExtensions.containsAll(expectedSettings));
    }

    /**
     * Test supported cases in remote indexing.
     */
    @SneakyThrows
    public void testSupportsRemoteIndexBuild() {
        KNNEngine Faiss = KNNEngine.FAISS;
        KNNEngine Lucene = KNNEngine.LUCENE;

        // Faiss
        // FP32, FP16
        assertTrue(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsFP32()));
        assertTrue(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsFP16()));

        // Byte
        assertTrue(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsByte(METHOD_HNSW)));

        // Pure Binary
        assertTrue(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsBinary(METHOD_HNSW)));

        // Quantized case
        assertTrue(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsQuantized()));

        // IVF all must fail
        assertFalse(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsByte(METHOD_IVF)));
        assertFalse(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsBinary(METHOD_IVF)));
        assertFalse(Faiss.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsIVF()));

        // Lucene
        assertFalse(Lucene.supportsRemoteIndexBuild(createMockKnnLibraryIndexingContextParamsFP32()));
    }

    @SneakyThrows
    public void testCreateRemoteIndexingParameters_Success() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        RemoteIndexParameters result = method.createRemoteIndexingParameters(
            createMockKnnLibraryIndexingContextParamsFP32().getLibraryParameters()
        );

        assertNotNull(result);
        assertTrue(result instanceof RemoteFaissHNSWIndexParameters);

        RemoteFaissHNSWIndexParameters hnswParams = (RemoteFaissHNSWIndexParameters) result;

        assertEquals(METHOD_HNSW, hnswParams.getAlgorithm());
        assertEquals(L2.getValue(), hnswParams.getSpaceType());
        assertEquals(94, hnswParams.getEfConstruction());
        assertEquals(89, hnswParams.getEfSearch());
        assertEquals(14, hnswParams.getM());
    }

    @SneakyThrows
    public void testCreateRemoteIndexingParameters_CosineSpaceType() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNLibraryIndexingContext knnLibraryIndexingContext = createMockKnnLibraryIndexingContextParamsCosine();

        RemoteIndexParameters result = method.createRemoteIndexingParameters(knnLibraryIndexingContext.getLibraryParameters());

        assertNotNull(result);
        assertTrue(result instanceof RemoteFaissHNSWIndexParameters);

        RemoteFaissHNSWIndexParameters hnswParams = (RemoteFaissHNSWIndexParameters) result;

        // Test that cosine space type is converted to inner product space type for Faiss
        assertEquals(SpaceType.INNER_PRODUCT.getValue(), hnswParams.getSpaceType());
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsFP32() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 89)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 94)
            .field(METHOD_PARAMETER_M, 14)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_FLAT)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsFP16() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 89)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 94)
            .field(METHOD_PARAMETER_M, 14)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsByte(final String method) {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BYTE)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, method)
            .field(METHOD_PARAMETER_SPACE_TYPE, INNER_PRODUCT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 28)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_FLAT)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsIVF() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(METHOD_PARAMETER_SPACE_TYPE, INNER_PRODUCT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 28)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_FLAT)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsBinary(final String method) {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BINARY)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, method)
            .field(METHOD_PARAMETER_SPACE_TYPE, INNER_PRODUCT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 28)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_FLAT)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsQuantized() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, INNER_PRODUCT.getValue())
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 28)
            .startObject(METHOD_ENCODER_PARAMETER)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNLibraryIndexingContext mockContext = mock(KNNLibraryIndexingContext.class);
        when(mockContext.getLibraryParameters()).thenReturn(in);
        return mockContext;
    }

    @SneakyThrows
    private KNNLibraryIndexingContext createMockKnnLibraryIndexingContextParamsCosine() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BINARY)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, COSINESIMIL.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_SEARCH, 24)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 28)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_FLAT)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(in);
        return Faiss.INSTANCE.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext);
    }
}
