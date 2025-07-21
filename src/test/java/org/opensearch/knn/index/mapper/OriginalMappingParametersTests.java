/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Collections;

public class OriginalMappingParametersTests extends KNNTestCase {

    public void testIsLegacy() {
        assertTrue(
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                123,
                null,
                null,
                null,
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            ).isLegacyMapping()
        );
        assertFalse(
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                123,
                null,
                null,
                null,
                "model-id",
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            ).isLegacyMapping()
        );
        assertFalse(
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                123,
                null,
                Mode.ON_DISK.getName(),
                null,
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            ).isLegacyMapping()
        );
        assertFalse(
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                123,
                null,
                null,
                CompressionLevel.x2.getName(),
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            ).isLegacyMapping()
        );
        assertFalse(
            new OriginalMappingParameters(
                VectorDataType.DEFAULT,
                123,
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.L2, new MethodComponentContext(null, Collections.emptyMap())),
                null,
                null,
                null,
                SpaceType.UNDEFINED.getValue(),
                KNNEngine.UNDEFINED.getName()
            ).isLegacyMapping()
        );
    }

}
