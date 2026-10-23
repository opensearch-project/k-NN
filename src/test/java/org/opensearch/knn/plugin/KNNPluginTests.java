/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.xcontent.yaml.YamlXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.transport.RemoveModelFromCacheAction;
import org.opensearch.knn.plugin.transport.UpdateModelGraveyardAction;
import org.opensearch.knn.plugin.transport.UpdateModelMetadataAction;
import org.opensearch.knn.search.processor.mmr.MMRKnnQueryTransformer;
import org.opensearch.knn.search.processor.mmr.MMRQueryTransformer;
import org.opensearch.plugins.ExtensiblePlugin;

import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

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

    public void testPluginAdditionalPermissionsGrantTheModelActionsModelDaoRunsAsThePluginSubject() throws Exception {
        // The security plugin reads this file through the plugin's own class loader and grants only the
        // literal action names listed, so a renamed action would fail closed without this check.
        try (
            InputStream in = Objects.requireNonNull(
                KNNPlugin.class.getClassLoader().getResourceAsStream("plugin-additional-permissions.yml"),
                "plugin-additional-permissions.yml must be on the plugin classpath"
            );
            XContentParser parser = YamlXContent.yamlXContent.createParser(
                NamedXContentRegistry.EMPTY,
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                in
            )
        ) {
            Map<String, Object> permissions = parser.map();
            assertEquals(Set.of("cluster_permissions"), permissions.keySet());
            assertEquals(
                Set.of(UpdateModelMetadataAction.NAME, UpdateModelGraveyardAction.NAME, RemoveModelFromCacheAction.NAME),
                Set.copyOf((List<?>) permissions.get("cluster_permissions"))
            );
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, MMRQueryTransformer<?>> getMmrQueryTransformers() throws Exception {
        Field field = KNNPlugin.class.getDeclaredField("mmrQueryTransformers");
        field.setAccessible(true);
        return (Map<String, MMRQueryTransformer<?>>) field.get(knnPlugin);
    }
}
