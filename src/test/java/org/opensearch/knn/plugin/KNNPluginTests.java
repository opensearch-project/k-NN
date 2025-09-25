/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.search.processor.mmr.MMRKnnQueryTransformer;
import org.opensearch.knn.search.processor.mmr.MMRQueryTransformer;
import org.opensearch.plugins.ExtensiblePlugin;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KNNPluginTests extends KNNTestCase {
    private KNNPlugin knnPlugin;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        knnPlugin = new KNNPlugin();
    }

    public void testLoadExtensions_whenSuccess() throws Exception {
        MMRQueryTransformer<?> transformer = mock(MMRQueryTransformer.class);
        when(transformer.getQueryName()).thenReturn("test_query");

        ExtensiblePlugin.ExtensionLoader loader = new ExtensiblePlugin.ExtensionLoader() {
            @SuppressWarnings("unchecked")
            @Override
            public <T> List<T> loadExtensions(Class<T> extensionPointType) {
                if (extensionPointType.equals(MMRQueryTransformer.class)) {
                    return (List<T>) List.of(transformer);
                }
                return List.of();
            }
        };

        knnPlugin.loadExtensions(loader);

        Map<String, MMRQueryTransformer<?>> map = getMmrQueryTransformers();
        assertEquals(transformer, map.get("test_query"));
        assertTrue(map.get(KNNQueryBuilder.NAME) instanceof MMRKnnQueryTransformer);
    }

    public void testLoadExtensions_whenDuplicatedThenException() throws Exception {
        MMRQueryTransformer<?> transformerA = mock(MMRQueryTransformer.class);
        when(transformerA.getQueryName()).thenReturn("test_query");
        MMRQueryTransformer<?> transformerB = mock(MMRQueryTransformer.class);
        when(transformerB.getQueryName()).thenReturn("test_query");

        ExtensiblePlugin.ExtensionLoader loader = new ExtensiblePlugin.ExtensionLoader() {
            @SuppressWarnings("unchecked")
            @Override
            public <T> List<T> loadExtensions(Class<T> extensionPointType) {
                if (extensionPointType.equals(MMRQueryTransformer.class)) {
                    return (List<T>) List.of(transformerA, transformerB);
                }
                return List.of();
            }
        };

        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> knnPlugin.loadExtensions(loader));

        String expectedError = "Already load the MMR query transformer";
        assertTrue(exception.getMessage().contains(expectedError));
    }

    @SuppressWarnings("unchecked")
    private Map<String, MMRQueryTransformer<?>> getMmrQueryTransformers() throws Exception {
        Field field = KNNPlugin.class.getDeclaredField("mmrQueryTransformers");
        field.setAccessible(true);
        return (Map<String, MMRQueryTransformer<?>>) field.get(knnPlugin);
    }
}
