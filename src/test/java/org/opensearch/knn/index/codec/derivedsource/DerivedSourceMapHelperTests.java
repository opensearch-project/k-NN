/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.SneakyThrows;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DerivedSourceMapHelperTests extends KNNTestCase {

    @SneakyThrows
    public void testFilterFields() {
        // Basic filter
        String[] filterFields = new String[] { "field1", "field2", "field3" };
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field("field1", "value")
            .field("field2", "value")
            .field("field3", "value")
            .endObject();
        Map<String, Object> source = DerivedSourceMapHelper.filterFields(filterFields, xContentToMap(builder));
        assertEquals(Collections.emptyMap(), source);
        source = DerivedSourceMapHelper.filterFields(new String[] { "field1" }, xContentToMap(builder));
        assertTrue(DerivedSourceMapHelper.fieldExists(source, "field2"));
        assertTrue(DerivedSourceMapHelper.fieldExists(source, "field3"));
        assertFalse(DerivedSourceMapHelper.fieldExists(source, "field1"));

        // Object case
        filterFields = new String[] { "level1.level2.test", "field.nonexist" };
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("level1")
            .startObject("level2")
            .field("test", "test")
            .endObject()
            .endObject()
            .endObject();
        source = DerivedSourceMapHelper.filterFields(filterFields, xContentToMap(builder));
        assertEquals(Map.of("level1", Map.of("level2", Collections.emptyMap())), source);

        // Nested case. In this case, if filtering out a value leads it to be an empty map, then it will be removed
        // from the array
        filterFields = new String[] { "nested.deep.value", "nested.vector" };
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startArray("nested")
            .startObject()
            .field("text", "text1")
            .startArray("deep")
            .startObject()
            .field("value", "text2")
            .endObject()
            .endArray()
            .endObject()
            .startObject()
            .field("vector", "text1")
            .endObject()
            .startObject()
            .field("text", "text1")
            .field("vector", "text1")
            .endObject()
            .endArray()
            .endObject();
        source = DerivedSourceMapHelper.filterFields(filterFields, xContentToMap(builder));
        assertTrue(source.containsKey("nested"));
        Object nested = source.get("nested");
        assertTrue(nested instanceof List<?>);
        List<?> nestedList = (List<?>) nested;
        assertEquals(2, nestedList.size());
        assertTrue(nestedList.get(0) instanceof Map);
        Map<String, Object> nestedMap = (Map<String, Object>) nestedList.get(0);
        assertTrue(nestedMap.containsKey("deep"));
        Object deep = nestedMap.get("deep");
        assertTrue(deep instanceof List<?>);
        assertEquals(0, ((List<?>) deep).size());
    }

    @SneakyThrows
    public void testFieldExists() {
        // Basic, flat validation
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("field", "value").endObject();
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field"));
        assertFalse(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "non-existent-field"));

        // Object type
        builder = XContentFactory.jsonBuilder().startObject().startObject("field").field("test", "test").endObject().endObject();
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field.test"));
        assertFalse(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "test"));

        // Complex nested object
        builder = XContentFactory.jsonBuilder()
            .startObject()
            .startArray("nested1")
            .startObject()
            .field("field1", "test")
            .endObject()
            .startObject()
            .field("field1", "test")
            .field("field2", "test")
            .endObject()
            .startObject()
            .field("field3", "test")
            .endObject()
            .endArray()
            .startArray("nested2")
            .startObject()
            .field("field1", "test")
            .endObject()
            .startObject()
            .field("field1", "test")
            .field("field2", "test")
            .endObject()
            .startObject()
            .field("field3", "test")
            .endObject()
            .endArray()
            .field("notnested", "test")
            .endObject();
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested1"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested2"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "notnested"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested1.field1"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested1.field2"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested1.field3"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested2.field1"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested2.field2"));
        assertTrue(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "nested2.field3"));
        assertFalse(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field1"));
        assertFalse(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field2"));
        assertFalse(DerivedSourceMapHelper.fieldExists(xContentToMap(builder), "field3"));

        // Null test
        Map<String, Object> map = new HashMap<>();
        map.put("test", null);
        assertTrue(DerivedSourceMapHelper.fieldExists(map, "test"));
    }

    @SneakyThrows
    public void testInjectObject() {
        // Base case
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().field("field1", "value").field("field3", "value").endObject();
        Map<String, Object> source = xContentToMap(builder);
        DerivedSourceMapHelper.injectObject(source, "test", "field2");
        assertEquals(Map.of("field1", "value", "field2", "test", "field3", "value"), source);

        // Nested cases
        source = xContentToMap(builder);
        DerivedSourceMapHelper.injectObject(source, "test", "field2.nested");
        assertEquals(Map.of("field1", "value", "field2", Map.of("nested", "test"), "field3", "value"), source);

        source = xContentToMap(builder);
        DerivedSourceMapHelper.injectObject(source, "test", "field2.nested1.nested2");
        assertEquals(Map.of("field1", "value", "field2", Map.of("nested1", Map.of("nested2", "test")), "field3", "value"), source);
    }

    private Map<String, Object> xContentToMap(XContentBuilder xContentBuilder) {
        return XContentHelper.convertToMap(BytesReference.bytes(xContentBuilder), true, xContentBuilder.contentType()).v2();
    }
}
