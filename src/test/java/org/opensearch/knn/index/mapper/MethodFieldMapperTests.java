/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import junit.framework.TestCase;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.util.Collections;

public class MethodFieldMapperTests extends TestCase {
    public void testMethodFieldMapper_whenVectorDataTypeIsGiven_thenSetItInFieldType() {
        KNNVectorFieldMapper.KNNVectorFieldType mappedFieldType = new KNNVectorFieldMapper.KNNVectorFieldType(
            "testField",
            Collections.emptyMap(),
            1,
            VectorDataType.BINARY,
            SpaceType.HAMMING
        );
        MethodFieldMapper mappers = new MethodFieldMapper(
            "simpleName",
            mappedFieldType,
            null,
            new FieldMapper.CopyTo.Builder().build(),
            KNNVectorFieldMapper.Defaults.IGNORE_MALFORMED,
            true,
            true,
            KNNMethodContext.getDefault()
        );
        assertEquals(VectorDataType.BINARY, mappers.fieldType().vectorDataType);
    }
}
