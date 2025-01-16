/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

public class VectorTransformerFactoryTests extends KNNTestCase {

    public void testAllSpaceTypes_withFaiss() {
        for (SpaceType spaceType : SpaceType.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.FAISS, spaceType);
            validateTransformer(spaceType, KNNEngine.FAISS, transformer);
        }
    }

    public void testAllEngines_withCosine() {
        for (KNNEngine engine : KNNEngine.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(engine, SpaceType.COSINESIMIL);
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
                VectorTransformerFactory.getVectorTransformer(),
                transformer
            );
        }
    }
}
