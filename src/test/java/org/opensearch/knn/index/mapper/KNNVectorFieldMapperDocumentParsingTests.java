/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexableField;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.mapper.SourceToParse;
import org.opensearch.knn.KNNSingleNodeTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Parses whole documents through the real {@link DocumentMapper}, so that the position the vector parser leaves the
 * {@link org.opensearch.core.xcontent.XContentParser} in is checked against what the document parser expects.
 */
public class KNNVectorFieldMapperDocumentParsingTests extends KNNSingleNodeTestCase {

    private static final String INDEX_NAME = "test-index";
    private static final String[] VECTOR_FIELDS = { "float_vector", "byte_vector" };

    public void testParse_whenOneDimensionalVectorIsBareNumberAndLastField_thenParsedAsOneElementArray() throws IOException {
        DocumentMapper documentMapper = createIndexWithOneDimensionalVectors();
        for (String field : VECTOR_FIELDS) {
            ParsedDocument expected = parse(documentMapper, "{\"id\": 1, \"" + field + "\": [7]}");
            ParsedDocument actual = parse(documentMapper, "{\"id\": 1, \"" + field + "\": 7}");
            assertEquals(describeFields(expected, field, "id"), describeFields(actual, field, "id"));
        }
    }

    public void testParse_whenOneDimensionalVectorIsBareNumberAndFollowedByField_thenFollowingFieldParsed() throws IOException {
        DocumentMapper documentMapper = createIndexWithOneDimensionalVectors();
        for (String field : VECTOR_FIELDS) {
            ParsedDocument expected = parse(documentMapper, "{\"" + field + "\": [7], \"id\": 1}");
            ParsedDocument actual = parse(documentMapper, "{\"" + field + "\": 7, \"id\": 1}");
            assertEquals(describeFields(expected, field, "id"), describeFields(actual, field, "id"));
            assertNotEquals(0, actual.rootDoc().getFields("id").length);
        }
    }

    public void testParse_whenOneDimensionalVectorIsBareNumberInsideObject_thenObjectClosedBeforeNextField() throws IOException {
        DocumentMapper documentMapper = createIndexWithOneDimensionalVectors();
        for (String field : VECTOR_FIELDS) {
            String objectField = "object." + field;
            ParsedDocument expected = parse(documentMapper, "{\"object\": {\"" + field + "\": [7]}, \"id\": 1}");
            ParsedDocument actual = parse(documentMapper, "{\"object\": {\"" + field + "\": 7}, \"id\": 1}");
            assertEquals(describeFields(expected, objectField, "id", "object.id"), describeFields(actual, objectField, "id", "object.id"));
            assertNotEquals(0, actual.rootDoc().getFields("id").length);
        }
    }

    private DocumentMapper createIndexWithOneDimensionalVectors() throws IOException {
        XContentBuilder mapping = XContentFactory.jsonBuilder().startObject().startObject("properties");
        addOneDimensionalVectorAndIdProperties(mapping);
        mapping.startObject("object").startObject("properties");
        addOneDimensionalVectorAndIdProperties(mapping);
        mapping.endObject().endObject();
        mapping.endObject().endObject();

        return createIndex(INDEX_NAME, getKNNDefaultIndexSettingsBuildsGraphAlways(), mapping).mapperService().documentMapper();
    }

    private static void addOneDimensionalVectorAndIdProperties(XContentBuilder properties) throws IOException {
        properties.startObject("float_vector").field("type", "knn_vector").field("dimension", 1).endObject();
        properties.startObject("byte_vector").field("type", "knn_vector").field("dimension", 1).field("data_type", "byte").endObject();
        properties.startObject("id").field("type", "integer").endObject();
    }

    private static ParsedDocument parse(DocumentMapper documentMapper, String source) {
        return documentMapper.parse(new SourceToParse(INDEX_NAME, "1", new BytesArray(source), MediaTypeRegistry.JSON));
    }

    private static List<String> describeFields(ParsedDocument document, String... fieldNames) {
        List<String> description = new ArrayList<>();
        for (String fieldName : fieldNames) {
            for (IndexableField field : document.rootDoc().getFields(fieldName)) {
                description.add(describeField(field));
            }
        }
        return description;
    }

    private static String describeField(IndexableField field) {
        if (field instanceof KnnFloatVectorField floatVectorField) {
            return field.name() + Arrays.toString(floatVectorField.vectorValue());
        }
        if (field instanceof KnnByteVectorField byteVectorField) {
            return field.name() + Arrays.toString(byteVectorField.vectorValue());
        }
        return field.toString();
    }
}
