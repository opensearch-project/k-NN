/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import junit.framework.TestCase;
import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.mockito.Mockito;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;

import java.util.Collections;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

public class FieldInfoExtractorTests extends TestCase {
    @SneakyThrows
    public void testGetIndexDescription_whenNoDescription_thenReturnNull() {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.attributes()).thenReturn(Collections.emptyMap(), Map.of(KNNConstants.PARAMETERS, "{}"));
        assertNull(FieldInfoExtractor.getIndexDescription(fieldInfo));
        assertNull(FieldInfoExtractor.getIndexDescription(fieldInfo));
    }

    @SneakyThrows
    public void testGetIndexDescription_whenDescriptionExist_thenReturnIndexDescription() {
        String indexDescription = "HNSW";
        XContentBuilder parameters = XContentFactory.jsonBuilder()
            .startObject()
            .field(INDEX_DESCRIPTION_PARAMETER, indexDescription)
            .endObject();
        FieldInfo fieldInfo = mock(FieldInfo.class);
        Mockito.when(fieldInfo.attributes()).thenReturn(Map.of(KNNConstants.PARAMETERS, parameters.toString()));
        assertEquals(indexDescription, FieldInfoExtractor.getIndexDescription(fieldInfo));
    }
}
