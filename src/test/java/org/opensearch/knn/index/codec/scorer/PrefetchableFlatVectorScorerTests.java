/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;

/**
 * Tests that {@link PrefetchableFlatVectorScorer} overrides all methods from {@link
 * org.apache.lucene.codecs.hnsw.FlatVectorsScorer}.
 */
public class PrefetchableFlatVectorScorerTests extends KNNTestCase {

    @Test
    public void testDeclaredMethodsOverridden() {
        assertAllMethodsOverridden(FlatVectorsScorer.class, PrefetchableFlatVectorScorer.class);
        assertAllMethodsOverridden(PrefetchableRandomVectorScorer.class.getSuperclass(), PrefetchableRandomVectorScorer.class);
    }
}
