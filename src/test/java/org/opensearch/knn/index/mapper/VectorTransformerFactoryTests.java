/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class VectorTransformerFactoryTests extends KNNTestCase {

    public void testAllSpaceTypes_withFaiss() {
        for (SpaceType spaceType : SpaceType.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.FAISS, spaceType, null);
            validateTransformer(spaceType, KNNEngine.FAISS, transformer);
        }
    }

    public void testAllEngines_withCosine() {
        for (KNNEngine engine : KNNEngine.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(engine, SpaceType.COSINESIMIL, null);
            validateTransformer(SpaceType.COSINESIMIL, engine, transformer);
        }
    }

    public void testLuceneCosine_withFlatMethod_returnsNormalizer() {
        MethodComponentContext flatContext = new MethodComponentContext(METHOD_FLAT, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.COSINESIMIL, flatContext);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testLuceneCosine_withSQOneBit_returnsNormalizer() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 1));
        MethodComponentContext hnswCtx = new MethodComponentContext(
            METHOD_HNSW,
            new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.COSINESIMIL, hnswCtx);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testLuceneCosine_withSQSevenBit_returnsNoop() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 7));
        MethodComponentContext hnswCtx = new MethodComponentContext(
            METHOD_HNSW,
            new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.COSINESIMIL, hnswCtx);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneCosine_withHnswNoEncoder_returnsNoop() {
        MethodComponentContext hnswCtx = new MethodComponentContext(METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.COSINESIMIL, hnswCtx);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneNonCosine_withFlatMethod_returnsNoop() {
        MethodComponentContext flatContext = new MethodComponentContext(METHOD_FLAT, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.L2, flatContext);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneCosine_withNullMethodComponentContext_returnsNoop() {
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, SpaceType.COSINESIMIL, null);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testFaissCosine_withMethodComponentContext_returnsNormalizer() {
        MethodComponentContext hnswCtx = new MethodComponentContext(METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.FAISS, SpaceType.COSINESIMIL, hnswCtx);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_faissCosine_returnsNormalizer() throws Exception {
        FieldInfo fieldInfo = buildFieldInfoWithParameters(KNNEngine.FAISS, SpaceType.COSINESIMIL, METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_faissL2_returnsNoop() throws Exception {
        FieldInfo fieldInfo = buildFieldInfoWithParameters(KNNEngine.FAISS, SpaceType.L2, METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_missingAttributes_returnsNoop() {
        FieldInfo fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("test_field").build();
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_luceneCosineFlat_returnsNormalizer() throws Exception {
        FieldInfo fieldInfo = buildFieldInfoWithParameters(KNNEngine.LUCENE, SpaceType.COSINESIMIL, METHOD_FLAT, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_luceneCosineSQOneBit_returnsNormalizer() throws Exception {
        // Build a nested encoder as a plain Map<String, Object> because the test helper JSON-serializes
        // the library parameters map; only plain Maps (not MethodComponentContext instances) serialize
        // correctly. The parser in VectorTransformerFactory reconstructs MethodComponentContext from the
        // nested Map.
        Map<String, Object> encoderMap = new HashMap<>();
        encoderMap.put(KNNConstants.NAME, ENCODER_SQ);
        encoderMap.put(KNNConstants.PARAMETERS, Map.of(LUCENE_SQ_BITS, 1));
        Map<String, Object> hnswParams = new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, encoderMap));
        FieldInfo fieldInfo = buildFieldInfoWithParameters(KNNEngine.LUCENE, SpaceType.COSINESIMIL, METHOD_HNSW, hnswParams);
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testGetVectorTransformer_fromFieldInfo_luceneCosineHnswNoEncoder_returnsNoop() throws Exception {
        FieldInfo fieldInfo = buildFieldInfoWithParameters(KNNEngine.LUCENE, SpaceType.COSINESIMIL, METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(fieldInfo, true);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    /**
     * Builds a FieldInfo whose attributes match what EngineFieldMapper writes, so that the test exercises
     * the same PARAMETERS JSON format used at runtime.
     */
    private static FieldInfo buildFieldInfoWithParameters(
        KNNEngine engine,
        SpaceType spaceType,
        String methodName,
        Map<String, Object> methodParameters
    ) throws Exception {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            engine,
            spaceType,
            new MethodComponentContext(methodName, methodParameters)
        );
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(org.opensearch.Version.CURRENT)
            .build();
        String parametersString = XContentFactory.jsonBuilder()
            .map(engine.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext).getLibraryParameters())
            .toString();
        return KNNCodecTestUtil.FieldInfoBuilder.builder("test_field")
            .addAttribute(KNNConstants.KNN_ENGINE, engine.getName())
            .addAttribute(KNNConstants.SPACE_TYPE, spaceType.getValue())
            .addAttribute(KNNConstants.PARAMETERS, parametersString)
            .build();
    }

    private static void validateTransformer(SpaceType spaceType, KNNEngine engine, VectorTransformer transformer) {
        if (spaceType == SpaceType.COSINESIMIL && engine == KNNEngine.FAISS) {
            assertTrue(
                "Should return NormalizeVectorTransformer for FAISS with " + spaceType,
                transformer instanceof NormalizeVectorTransformer
            );
        } else {
            assertSame(
                "Should return NOOP transformer for " + engine + " with COSINESIMIL",
                VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER,
                transformer
            );
        }
    }
}
