/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.SneakyThrows;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import static org.opensearch.Version.CURRENT;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.KNNSettings.KNN_SPACE_TYPE;

public class SpaceTypeResolverTests extends KNNTestCase {

    private static final SpaceTypeResolver SPACE_TYPE_RESOLVER = SpaceTypeResolver.INSTANCE;
    private static final Settings DONT_CARE_SETTINGS = null;
    private static final VectorDataType DONT_CARE_VECTOR_DATA = null;

    private void assertResolveSpaceType(
        KNNMethodContext knnMethodContext,
        String topLevelSpaceTypeString,
        Settings indexSettings,
        VectorDataType vectorDataType,
        SpaceType expectedSpaceType
    ) {
        assertEquals(
            expectedSpaceType,
            SPACE_TYPE_RESOLVER.resolveSpaceType(knnMethodContext, topLevelSpaceTypeString, indexSettings, vectorDataType)
        );
    }

    public void testResolveSpaceType_whenNoConfigProvided_thenFallbackToVectorDataType() {
        final Settings emptySettings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        final Settings settings = Settings.builder()
            .put(settings(CURRENT).build())
            .put(KNN_SPACE_TYPE, SpaceType.L2.getValue())
            .put(KNN_INDEX, true)
            .build();

        final KNNMethodContext methodContext = new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.COSINESIMIL, MethodComponentContext.EMPTY);
        final KNNMethodContext emptyMethodContext = new KNNMethodContext(
            KNNEngine.DEFAULT,
            SpaceType.UNDEFINED,
            MethodComponentContext.EMPTY
        );
        final KNNMethodContext nullMethodContext = null;

        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            settings,
            VectorDataType.BYTE,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            settings,
            VectorDataType.FLOAT,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            settings,
            VectorDataType.BINARY,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BYTE,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.FLOAT,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(
            methodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BINARY,
            methodContext.getSpaceType()
        );
        assertResolveSpaceType(methodContext, "", settings, VectorDataType.BYTE, methodContext.getSpaceType());
        assertResolveSpaceType(methodContext, "", settings, VectorDataType.FLOAT, methodContext.getSpaceType());
        assertResolveSpaceType(methodContext, "", settings, VectorDataType.BINARY, methodContext.getSpaceType());
        assertResolveSpaceType(methodContext, "", emptySettings, VectorDataType.BYTE, methodContext.getSpaceType());
        assertResolveSpaceType(methodContext, "", emptySettings, VectorDataType.FLOAT, methodContext.getSpaceType());
        assertResolveSpaceType(methodContext, "", emptySettings, VectorDataType.BINARY, methodContext.getSpaceType());
        assertResolveSpaceType(emptyMethodContext, SpaceType.COSINESIMIL.getValue(), settings, VectorDataType.BYTE, SpaceType.COSINESIMIL);
        assertResolveSpaceType(emptyMethodContext, SpaceType.COSINESIMIL.getValue(), settings, VectorDataType.FLOAT, SpaceType.COSINESIMIL);
        assertResolveSpaceType(
            emptyMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            settings,
            VectorDataType.BINARY,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            emptyMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BYTE,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            emptyMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.FLOAT,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            emptyMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BINARY,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            emptyMethodContext,
            "",
            settings,
            VectorDataType.BYTE,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(
            emptyMethodContext,
            "",
            settings,
            VectorDataType.FLOAT,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(
            emptyMethodContext,
            "",
            settings,
            VectorDataType.BINARY,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(emptyMethodContext, "", emptySettings, VectorDataType.BYTE, SpaceType.DEFAULT);
        assertResolveSpaceType(emptyMethodContext, "", emptySettings, VectorDataType.FLOAT, SpaceType.DEFAULT);
        assertResolveSpaceType(emptyMethodContext, "", emptySettings, VectorDataType.BINARY, SpaceType.DEFAULT_BINARY);
        assertResolveSpaceType(nullMethodContext, SpaceType.COSINESIMIL.getValue(), settings, VectorDataType.BYTE, SpaceType.COSINESIMIL);
        assertResolveSpaceType(nullMethodContext, SpaceType.COSINESIMIL.getValue(), settings, VectorDataType.FLOAT, SpaceType.COSINESIMIL);
        assertResolveSpaceType(nullMethodContext, SpaceType.COSINESIMIL.getValue(), settings, VectorDataType.BINARY, SpaceType.COSINESIMIL);
        assertResolveSpaceType(
            nullMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BYTE,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            nullMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.FLOAT,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            nullMethodContext,
            SpaceType.COSINESIMIL.getValue(),
            emptySettings,
            VectorDataType.BINARY,
            SpaceType.COSINESIMIL
        );
        assertResolveSpaceType(
            nullMethodContext,
            "",
            settings,
            VectorDataType.BYTE,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(
            nullMethodContext,
            "",
            settings,
            VectorDataType.FLOAT,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(
            nullMethodContext,
            "",
            settings,
            VectorDataType.BINARY,
            SpaceType.getSpace(settings.get(KNNSettings.INDEX_KNN_SPACE_TYPE.getKey()))
        );
        assertResolveSpaceType(nullMethodContext, "", emptySettings, VectorDataType.BYTE, SpaceType.DEFAULT);
        assertResolveSpaceType(nullMethodContext, "", emptySettings, VectorDataType.FLOAT, SpaceType.DEFAULT);
        assertResolveSpaceType(nullMethodContext, "", emptySettings, VectorDataType.BINARY, SpaceType.DEFAULT_BINARY);
    }

    @SneakyThrows
    public void testResolveSpaceType_whenMethodSpaceTypeAndTopLevelSpecified_thenThrowIfConflict() {
        expectThrows(
            MapperParsingException.class,
            () -> SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L2, MethodComponentContext.EMPTY),
                SpaceType.INNER_PRODUCT.getValue(),
                DONT_CARE_SETTINGS,
                DONT_CARE_VECTOR_DATA
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                SpaceType.DEFAULT.getValue(),
                DONT_CARE_SETTINGS,
                DONT_CARE_VECTOR_DATA
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                SpaceType.UNDEFINED.getValue(),
                DONT_CARE_SETTINGS,
                DONT_CARE_VECTOR_DATA
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                SpaceType.DEFAULT.getValue(),
                DONT_CARE_SETTINGS,
                DONT_CARE_VECTOR_DATA
            )
        );

        final Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        // method (undefined) -> top level (undefined) -> settings (undefined) -> Default Space Type
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                SpaceType.UNDEFINED.getValue(),
                settings,
                VectorDataType.BYTE
            )
        );

        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                SpaceType.UNDEFINED.getValue(),
                settings,
                VectorDataType.FLOAT
            )
        );

        assertEquals(
            SpaceType.DEFAULT_BINARY,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                SpaceType.UNDEFINED.getValue(),
                settings,
                VectorDataType.BINARY
            )
        );
    }

    @SneakyThrows
    public void testResolveSpaceType_whenSpaceTypeSpecifiedOnce_thenReturnValue() {
        assertEquals(
            SpaceType.L1,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L1, MethodComponentContext.EMPTY),
                "",
                null,
                null
            )
        );
        assertEquals(SpaceType.INNER_PRODUCT, SPACE_TYPE_RESOLVER.resolveSpaceType(null, SpaceType.INNER_PRODUCT.getValue(), null, null));
    }
}
