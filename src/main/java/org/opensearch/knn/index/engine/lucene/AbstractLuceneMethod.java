/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.mapper.VectorTransformer;
import org.opensearch.knn.index.mapper.VectorTransformerFactory;

import java.util.Set;

/**
 * Shared base for the Lucene engine's methods, mirroring {@link org.opensearch.knn.index.engine.faiss.AbstractFaissMethod}
 * so that both engines resolve their write-side vector transformer in the same place.
 */
public abstract class AbstractLuceneMethod extends AbstractKNNMethod {

    public AbstractLuceneMethod(MethodComponent methodComponent, Set<SpaceType> spaces, KNNLibrarySearchContext knnLibrarySearchContext) {
        super(methodComponent, spaces, knnLibrarySearchContext);
    }

    /**
     * Normalizes half_float cosine vectors on write: FP16_COSINE computes {@code (1 + dot) / 2}, which
     * equals cosine only for unit vectors. float32 is excluded - Lucene's quantized writer handles it.
     */
    @Override
    protected VectorTransformer getVectorTransformer(SpaceType spaceType, VectorDataType vectorDataType) {
        if (VectorDataType.HALF_FLOAT != vectorDataType) {
            return VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER;
        }
        return VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, spaceType, null, vectorDataType);
    }
}
