/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class KNNDynamicTemplateTypeHandlerTests extends KNNTestCase {

    private final KNNDynamicTemplateTypeHandler handler = new KNNDynamicTemplateTypeHandler();

    /** A supplier over a flat numeric array of the given length — get() yields a parser at START_ARRAY. */
    private FieldValueParserSupplier arraySupplier(int dimension) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dimension; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("0.1");
        }
        sb.append("]");
        return supplierOver(sb.toString());
    }

    /** A supplier over the given JSON value bytes; get() creates a parser positioned at the first token. */
    private FieldValueParserSupplier supplierOver(String json) {
        return new FieldValueParserSupplier(
            MediaTypeRegistry.JSON,
            NamedXContentRegistry.EMPTY,
            LoggingDeprecationHandler.INSTANCE,
            json.getBytes(java.nio.charset.StandardCharsets.UTF_8)
        );
    }

    /** A supplier whose get() throws — asserts the handler must not read the value for a complete config. */
    private FieldValueParserSupplier failingSupplier() {
        return FieldValueParserSupplier.withoutValue();
    }

    public void testInjectsDimensionFromArrayLength() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        handler.adjustMappingConfig(config, arraySupplier(128));
        assertEquals(128, config.get(KNNConstants.DIMENSION));
    }

    public void testInjectsTypeWhenAbsent() throws IOException {
        // Template with an empty mapping block ({}): handler injects knn_vector type, then dimension.
        Map<String, Object> config = new HashMap<>();
        handler.adjustMappingConfig(config, arraySupplier(256));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(256, config.get(KNNConstants.DIMENSION));
    }

    public void testDoesNotOverrideUserType() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", "some_other_type");
        config.put(KNNConstants.DIMENSION, 32);
        handler.adjustMappingConfig(config, failingSupplier());
        assertEquals("some_other_type", config.get("type"));
    }

    public void testDimensionPresentOpensNoParser() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 64);
        handler.adjustMappingConfig(config, failingSupplier());
        assertEquals(64, config.get(KNNConstants.DIMENSION));
    }

    public void testModelIdPresentOpensNoParserAndInjectsNoDimension() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        handler.adjustMappingConfig(config, failingSupplier());
        assertFalse("dimension must not be injected when a model supplies it", config.containsKey(KNNConstants.DIMENSION));
    }

    public void testIsConfigCompleteWithDimension() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 128);
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigCompleteWithModelId() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigIncompleteWithoutDimensionOrModel() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        assertFalse(handler.isConfigComplete(config));
    }

    public void testNonArrayValueInjectsNoDimension() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        // Field value is a string, not an array — handler must not infer a dimension.
        handler.adjustMappingConfig(config, supplierOver("\"not-an-array\""));
        assertFalse(config.containsKey(KNNConstants.DIMENSION));
    }
}
