/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.indices;

import lombok.SneakyThrows;
import org.opensearch.OpenSearchParseException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class ModelGraveyardTests extends OpenSearchTestCase {

    public void testAdd() {
        ModelGraveyard testModelGraveyard = new ModelGraveyard();
        String testModelId = "test-model-id";
        testModelGraveyard.add(testModelId);
        assertTrue(testModelGraveyard.contains(testModelId));
    }

    public void testRemove() {
        Set<String> modelIds = new HashSet<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);
        ModelGraveyard testModelGraveyard = new ModelGraveyard(modelIds);

        assertTrue(testModelGraveyard.contains(testModelId));
        testModelGraveyard.remove(testModelId);
        assertFalse(testModelGraveyard.contains(testModelId));
    }

    public void testContains() {
        Set<String> modelIds = new HashSet<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);

        ModelGraveyard testModelGraveyard = new ModelGraveyard(modelIds);
        assertTrue(testModelGraveyard.contains(testModelId));
    }

    public void testStreams() throws IOException {
        Set<String> modelIds = new HashSet<>();
        String testModelId = "test-model-id";
        modelIds.add(testModelId);
        ModelGraveyard testModelGraveyard = new ModelGraveyard(modelIds);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        testModelGraveyard.writeTo(streamOutput);

        ModelGraveyard testModelGraveyardCopy = new ModelGraveyard(streamOutput.bytes().streamInput());

        assertEquals(testModelGraveyard.size(), testModelGraveyardCopy.size());
        assertTrue(testModelGraveyard.contains(testModelId));
        assertTrue(testModelGraveyardCopy.contains(testModelId));
    }

    // Validating {model_ids: ["test-model-id1", "test-model-id2"]}
    @SneakyThrows
    public void testXContentBuilder_withModelIds_returnsModelGraveyardWithModelIds() {
        Set<String> modelIds = new HashSet<>();
        String testModelId1 = "test-model-id1";
        String testModelId2 = "test-model-id2";
        modelIds.add(testModelId1);
        modelIds.add(testModelId2);
        ModelGraveyard testModelGraveyard = new ModelGraveyard(modelIds);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        XContentBuilder builder = testModelGraveyard.toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
        builder.endObject();

        ModelGraveyard testModelGraveyard2 = ModelGraveyard.fromXContent(createParser(builder));
        assertEquals(2, testModelGraveyard2.size());
        assertTrue(testModelGraveyard2.contains(testModelId1));
        assertTrue(testModelGraveyard2.contains(testModelId2));
    }

    // Validating {model_ids:[]}
    @SneakyThrows
    public void testXContentBuilder_withoutModelIds_returnsModelGraveyardWithoutModelIds() {
        ModelGraveyard testModelGraveyard = new ModelGraveyard();
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        XContentBuilder builder = testModelGraveyard.toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
        builder.endObject();

        ModelGraveyard testModelGraveyard2 = ModelGraveyard.fromXContent(createParser(builder));
        assertEquals(0, testModelGraveyard2.size());
    }

    // Validating {test-model:"abcd"}
    @SneakyThrows
    public void testXContentBuilder_withWrongFieldName_throwsException() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.field("test-model");
        xContentBuilder.value("abcd");
        xContentBuilder.endObject();

        OpenSearchParseException ex = expectThrows(
            OpenSearchParseException.class,
            () -> ModelGraveyard.fromXContent(createParser(xContentBuilder))
        );
        assertTrue(ex.getMessage().contains("Expecting field model_ids but got test-model"));
    }

    // Validating {}
    @SneakyThrows
    public void testXContentBuilder_validateBackwardCompatibility_returnsEmptyModelGraveyardObject() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.endObject();

        ModelGraveyard testModelGraveyard = ModelGraveyard.fromXContent(createParser(xContentBuilder));
        assertEquals(0, testModelGraveyard.size());
    }

    // Validating null
    @SneakyThrows
    public void testXContentBuilder_withNull_throwsExceptionExpectingStartObject() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();

        OpenSearchParseException ex = expectThrows(
            OpenSearchParseException.class,
            () -> ModelGraveyard.fromXContent(createParser(xContentBuilder))
        );
        assertTrue(ex.getMessage().contains("Expecting token start of an object but got null"));
    }

    // Validating {model_ids:"abcd"}
    @SneakyThrows
    public void testXContentBuilder_withMissingStartArray_throwsExceptionExpectingStartArray() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.field("model_ids");
        xContentBuilder.value("abcd");
        xContentBuilder.endObject();

        OpenSearchParseException ex = expectThrows(
            OpenSearchParseException.class,
            () -> ModelGraveyard.fromXContent(createParser(xContentBuilder))
        );
        assertTrue(ex.getMessage().contains("Expecting token start of an array but got VALUE_STRING"));
    }

    // Validating {model_ids:["abcd"],model_ids_2:[]}
    @SneakyThrows
    public void testXContentBuilder_validateEndObject_throwsExceptionGotFieldName() {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.startArray("model_ids");
        xContentBuilder.value("abcd");
        xContentBuilder.endArray();
        xContentBuilder.startArray("model_ids_2");
        xContentBuilder.endArray();
        xContentBuilder.endObject();

        OpenSearchParseException ex = expectThrows(
            OpenSearchParseException.class,
            () -> ModelGraveyard.fromXContent(createParser(xContentBuilder))
        );
        assertTrue(ex.getMessage().contains("Expecting token end of an object but got FIELD_NAME"));
    }

    public void testDiffStreams() throws IOException {
        Set<String> added = new HashSet<>();
        Set<String> removed = new HashSet<>();
        String testModelId = "test-model-id";
        String testModelId1 = "test-model-id-1";
        added.add(testModelId);
        removed.add(testModelId1);

        ModelGraveyard modelGraveyardCurrent = new ModelGraveyard(added);
        ModelGraveyard modelGraveyardPrevious = new ModelGraveyard(removed);

        ModelGraveyard.ModelGraveyardDiff modelGraveyardDiff = new ModelGraveyard.ModelGraveyardDiff(
            modelGraveyardPrevious,
            modelGraveyardCurrent
        );
        assertEquals(added, modelGraveyardDiff.getAdded());
        assertEquals(removed, modelGraveyardDiff.getRemoved());

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        modelGraveyardDiff.writeTo(streamOutput);

        ModelGraveyard.ModelGraveyardDiff modelGraveyardDiffCopy = new ModelGraveyard.ModelGraveyardDiff(
            streamOutput.bytes().streamInput()
        );
        assertEquals(added, modelGraveyardDiffCopy.getAdded());
        assertEquals(removed, modelGraveyardDiffCopy.getRemoved());
    }

    public void testDiff() {

        // nothing will have been removed in previous object, and all entries in current object are new
        ModelGraveyard modelGraveyard1 = new ModelGraveyard();

        Set<String> modelIds = new HashSet<>();
        modelIds.add("1");
        modelIds.add("2");
        ModelGraveyard modelGraveyard2 = new ModelGraveyard(modelIds);

        ModelGraveyard.ModelGraveyardDiff diff1 = new ModelGraveyard.ModelGraveyardDiff(modelGraveyard1, modelGraveyard2);
        assertEquals(0, diff1.getRemoved().size());
        assertEquals(2, diff1.getAdded().size());

        ModelGraveyard updatedGraveyard1 = diff1.apply(modelGraveyard1);
        assertEquals(2, updatedGraveyard1.size());
        assertTrue(updatedGraveyard1.contains("1"));
        assertTrue(updatedGraveyard1.contains("2"));

        // nothing will have been added to current object, and all entries in previous object are removed
        ModelGraveyard modelGraveyard3 = new ModelGraveyard();
        ModelGraveyard.ModelGraveyardDiff diff2 = new ModelGraveyard.ModelGraveyardDiff(modelGraveyard2, modelGraveyard3);
        assertEquals(2, diff2.getRemoved().size());
        assertEquals(0, diff2.getAdded().size());

        ModelGraveyard updatedGraveyard2 = diff2.apply(modelGraveyard2);
        assertEquals(0, updatedGraveyard2.size());

        // some entries in previous object are removed and few entries are added to current object
        modelIds = new HashSet<>();
        modelIds.add("1");
        modelIds.add("3");
        modelIds.add("4");
        ModelGraveyard modelGraveyard4 = new ModelGraveyard(modelIds);

        ModelGraveyard.ModelGraveyardDiff diff3 = new ModelGraveyard.ModelGraveyardDiff(modelGraveyard2, modelGraveyard4);
        assertEquals(1, diff3.getRemoved().size());
        assertEquals(2, diff3.getAdded().size());

        ModelGraveyard updatedGraveyard3 = diff3.apply(modelGraveyard2);
        assertEquals(3, updatedGraveyard3.size());
        assertTrue(updatedGraveyard3.contains("1"));
        assertTrue(updatedGraveyard3.contains("3"));
        assertTrue(updatedGraveyard3.contains("4"));
        assertFalse(updatedGraveyard3.contains("2"));
    }

}
