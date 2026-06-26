/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.BuiltinKNNEngine;
import org.opensearch.knn.index.engine.KNNEngine;
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
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(BuiltinKNNEngine.FAISS, spaceType, null);
            validateTransformer(spaceType, BuiltinKNNEngine.FAISS, transformer);
        }
    }

    public void testAllEngines_withCosine() {
        for (KNNEngine engine : BuiltinKNNEngine.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(engine, SpaceType.COSINESIMIL, null);
            validateTransformer(SpaceType.COSINESIMIL, engine, transformer);
        }
    }

    public void testLuceneCosine_withFlatMethod_returnsNormalizer() {
        MethodComponentContext flatContext = new MethodComponentContext(METHOD_FLAT, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(
            BuiltinKNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            flatContext
        );
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testLuceneCosine_withSQOneBit_returnsNormalizer() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 1));
        MethodComponentContext hnswCtx = new MethodComponentContext(
            METHOD_HNSW,
            new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(
            BuiltinKNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            hnswCtx
        );
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    public void testLuceneCosine_withSQSevenBit_returnsNoop() {
        MethodComponentContext encoderCtx = new MethodComponentContext(ENCODER_SQ, Map.of(LUCENE_SQ_BITS, 7));
        MethodComponentContext hnswCtx = new MethodComponentContext(
            METHOD_HNSW,
            new HashMap<>(Map.of(METHOD_ENCODER_PARAMETER, encoderCtx))
        );
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(
            BuiltinKNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            hnswCtx
        );
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneCosine_withHnswNoEncoder_returnsNoop() {
        MethodComponentContext hnswCtx = new MethodComponentContext(METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(
            BuiltinKNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            hnswCtx
        );
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneNonCosine_withFlatMethod_returnsNoop() {
        MethodComponentContext flatContext = new MethodComponentContext(METHOD_FLAT, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(BuiltinKNNEngine.LUCENE, SpaceType.L2, flatContext);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testLuceneCosine_withNullMethodComponentContext_returnsNoop() {
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(BuiltinKNNEngine.LUCENE, SpaceType.COSINESIMIL, null);
        assertSame(VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER, transformer);
    }

    public void testFaissCosine_withMethodComponentContext_returnsNormalizer() {
        MethodComponentContext hnswCtx = new MethodComponentContext(METHOD_HNSW, new HashMap<>());
        VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(
            BuiltinKNNEngine.FAISS,
            SpaceType.COSINESIMIL,
            hnswCtx
        );
        assertTrue(transformer instanceof NormalizeVectorTransformer);
    }

    private static void validateTransformer(SpaceType spaceType, KNNEngine engine, VectorTransformer transformer) {
        if (spaceType == SpaceType.COSINESIMIL && engine == BuiltinKNNEngine.FAISS) {
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
