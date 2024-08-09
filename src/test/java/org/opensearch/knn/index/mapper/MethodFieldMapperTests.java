/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.util.Collections;

public class MethodFieldMapperTests extends KNNTestCase {
    public void testMethodFieldMapper_whenVectorDataTypeAndContextMismatch_thenThrow() {
        // Expect that we cannot create the mapper with an invalid field type
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        expectThrows(
            IllegalArgumentException.class,
            () -> MethodFieldMapper.createFieldMapper(
                "testField",
                "simpleName",
                Collections.emptyMap(),
                VectorDataType.BINARY,
                1,
                knnMethodContext,
                knnMethodContext,
                null,
                new FieldMapper.CopyTo.Builder().build(),
                KNNVectorFieldMapper.Defaults.IGNORE_MALFORMED,
                true,
                true,
                Version.CURRENT
            )
        );
    }
}
