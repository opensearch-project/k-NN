/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class VectorTransformerFactoryTests extends KNNTestCase {
    public void testAllSpaceTypes_withFaiss() {
        for (SpaceType spaceType : SpaceType.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.FAISS, spaceType);
            validateTransformer(spaceType, KNNEngine.FAISS, transformer);
        }
    }

    public void testAllEngines_withCosine() {
        // Test all engines with COSINESIMIL space type
        for (KNNEngine engine : KNNEngine.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(engine, SpaceType.COSINESIMIL);
            validateTransformer(SpaceType.COSINESIMIL, engine, transformer);
        }
    }

    public void testGetVectorTransformer_withNullContext() {
        // Test case for null context
        assertThrows(IllegalArgumentException.class, () -> VectorTransformerFactory.getVectorTransformer(null));
    }

    public void testAllSpaceTypes_usingContext_withFaiss() {
        for (SpaceType spaceType : SpaceType.values()) {
            KNNMethodContext context = mock(KNNMethodContext.class);
            when(context.getKnnEngine()).thenReturn(KNNEngine.FAISS);
            when(context.getSpaceType()).thenReturn(spaceType);
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(context);
            validateTransformer(spaceType, KNNEngine.FAISS, transformer);
        }
    }

    public void testAllEngines_usingContext_withCosine() {
        // Test all engines with COSINESIMIL space type
        for (KNNEngine engine : KNNEngine.values()) {
            KNNMethodContext context = mock(KNNMethodContext.class);
            when(context.getKnnEngine()).thenReturn(engine);
            when(context.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(context);
            validateTransformer(SpaceType.COSINESIMIL, engine, transformer);
        }
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
                VectorTransformer.NOOP_VECTOR_TRANSFORMER,
                transformer
            );
        }
    }
}
