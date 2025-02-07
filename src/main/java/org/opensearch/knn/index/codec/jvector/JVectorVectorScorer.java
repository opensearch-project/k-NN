/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

public class JVectorVectorScorer implements VectorScorer {
    private final JVectorFloatVectorValues floatVectorValues;
    private final VectorFloat<?> target;
    private final VectorSimilarityFunction similarityFunction;

    public JVectorVectorScorer(JVectorFloatVectorValues vectorValues, VectorFloat<?> target, VectorSimilarityFunction similarityFunction) {
        this.floatVectorValues = vectorValues;
        this.target = target;
        this.similarityFunction = similarityFunction;
    }

    @Override
    public float score() throws IOException {
        return similarityFunction.compare(target, floatVectorValues.vectorFloatValue());
    }

    @Override
    public DocIdSetIterator iterator() {
        return floatVectorValues;
    }
}
