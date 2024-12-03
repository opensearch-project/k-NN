/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.SneakyThrows;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

public class SpaceTypeResolverTests extends KNNTestCase {

    private static final SpaceTypeResolver SPACE_TYPE_RESOLVER = SpaceTypeResolver.INSTANCE;

    public void testResolveSpaceType_whenNoConfigProvided_thenFallbackToVectorDataType() {
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(null, VectorDataType.FLOAT, ""),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(null, VectorDataType.BYTE, ""),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                    VectorDataType.FLOAT,
                    ""
                ),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT_BINARY,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(null, VectorDataType.BINARY, ""),
                VectorDataType.BINARY
            )
        );
        assertEquals(
            SpaceType.DEFAULT_BINARY,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                    VectorDataType.BINARY,
                    ""
                ),
                VectorDataType.BINARY
            )
        );
    }

    @SneakyThrows
    public void testResolveSpaceType_whenMethodSpaceTypeAndTopLevelSpecified_thenThrowIfConflict() {
        expectThrows(
            MapperParsingException.class,
            () -> SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L2, MethodComponentContext.EMPTY),
                VectorDataType.FLOAT,
                SpaceType.INNER_PRODUCT.getValue()
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                    VectorDataType.FLOAT,
                    SpaceType.DEFAULT.getValue()
                ),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                    VectorDataType.FLOAT,
                    SpaceType.UNDEFINED.getValue()
                ),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                    VectorDataType.FLOAT,
                    SpaceType.DEFAULT.getValue()
                ),
                VectorDataType.FLOAT
            )
        );
        assertEquals(
            SpaceType.DEFAULT,
            SPACE_TYPE_RESOLVER.pickDefaultSpaceTypeWhenEmpty(
                SPACE_TYPE_RESOLVER.resolveSpaceType(
                    new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY),
                    VectorDataType.FLOAT,
                    SpaceType.UNDEFINED.getValue()
                ),
                VectorDataType.FLOAT
            )
        );
    }

    @SneakyThrows
    public void testResolveSpaceType_whenSpaceTypeSpecifiedOnce_thenReturnValue() {
        assertEquals(
            SpaceType.L1,
            SPACE_TYPE_RESOLVER.resolveSpaceType(
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L1, MethodComponentContext.EMPTY),
                VectorDataType.FLOAT,
                ""
            )
        );
        assertEquals(
            SpaceType.INNER_PRODUCT,
            SPACE_TYPE_RESOLVER.resolveSpaceType(null, VectorDataType.FLOAT, SpaceType.INNER_PRODUCT.getValue())
        );
    }
}
