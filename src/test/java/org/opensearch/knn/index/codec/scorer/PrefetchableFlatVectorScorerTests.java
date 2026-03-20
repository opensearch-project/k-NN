/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

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
        implTestDeclaredMethodsOverridden(FlatVectorsScorer.class, PrefetchableFlatVectorScorer.class);
        implTestDeclaredMethodsOverridden(
            PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer.class.getSuperclass(),
            PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer.class
        );
    }

    private void implTestDeclaredMethodsOverridden(Class<?> interfaceClass, Class<?> implClass) {
        for (final Method superClassMethod : interfaceClass.getDeclaredMethods()) {
            final int modifiers = superClassMethod.getModifiers();
            if (Modifier.isFinal(modifiers)) continue;
            if (Modifier.isStatic(modifiers)) continue;
            try {
                final Method subClassMethod = implClass.getDeclaredMethod(superClassMethod.getName(), superClassMethod.getParameterTypes());
                assertEquals("getReturnType() difference", superClassMethod.getReturnType(), subClassMethod.getReturnType());
            } catch (NoSuchMethodException e) {
                fail(implClass + " needs to override '" + superClassMethod + "'");
            }
        }
    }
}
