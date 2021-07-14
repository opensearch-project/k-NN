/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

public class ModelContextTests extends KNNTestCase {
    public void testGetModelId() {
        String modelId = "test-model-id";
        ModelContext modelContext = new ModelContext(modelId, KNNEngine.DEFAULT, SpaceType.DEFAULT);
        assertEquals(modelId, modelContext.getModelId());
    }

    public void testGetKNNEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelContext modelContext = new ModelContext(null, knnEngine, SpaceType.DEFAULT);
        assertEquals(knnEngine, modelContext.getKNNEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.DEFAULT;
        ModelContext modelContext = new ModelContext(null, KNNEngine.DEFAULT, spaceType);
        assertEquals(spaceType, modelContext.getSpaceType());
    }

    public void textToXContent() throws IOException {
        String modelId = "test-model";
        String spaceType = SpaceType.L2.getValue();
        String knnEngine = KNNEngine.DEFAULT.getName();
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(MODEL_ID, modelId)
                .field(KNN_ENGINE, knnEngine)
                .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
                .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        ModelContext modelContext = ModelContext.parse(in);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder = modelContext.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();

        Map<String, Object> out = xContentBuilderToMap(builder);
        assertEquals(modelId, out.get(MODEL_ID));
        assertEquals(knnEngine, out.get(KNN_ENGINE));
        assertEquals(spaceType, out.get(METHOD_PARAMETER_SPACE_TYPE));
    }

    public void testEquals() {
        String modelId1 = "test-model-1";
        String modelId2 = "test-model-2";
        SpaceType spaceType1 = SpaceType.L1;
        SpaceType spaceType2 = SpaceType.L2;
        KNNEngine knnEngine1= KNNEngine.FAISS;
        KNNEngine knnEngine2= KNNEngine.NMSLIB;

        ModelContext modelContext1 = new ModelContext(modelId1, knnEngine1, spaceType1);
        ModelContext modelContext2 = new ModelContext(modelId1, knnEngine1, spaceType1);
        ModelContext modelContext3 = new ModelContext(modelId2, knnEngine1, spaceType1);
        ModelContext modelContext4 = new ModelContext(modelId1, knnEngine2, spaceType1);
        ModelContext modelContext5 = new ModelContext(modelId1, knnEngine1, spaceType2);

        assertNotEquals(modelContext1, null);
        assertEquals(modelContext1, modelContext1);
        assertEquals(modelContext1, modelContext2);
        assertNotEquals(modelContext1, modelContext3);
        assertNotEquals(modelContext1, modelContext4);
        assertNotEquals(modelContext1, modelContext5);
    }

    public void testHashCode() {
        String modelId1 = "test-model-1";
        String modelId2 = "test-model-2";
        SpaceType spaceType1 = SpaceType.L1;
        SpaceType spaceType2 = SpaceType.L2;
        KNNEngine knnEngine1= KNNEngine.FAISS;
        KNNEngine knnEngine2= KNNEngine.NMSLIB;

        ModelContext modelContext1 = new ModelContext(modelId1, knnEngine1, spaceType1);
        ModelContext modelContext2 = new ModelContext(modelId1, knnEngine1, spaceType1);
        ModelContext modelContext3 = new ModelContext(modelId2, knnEngine1, spaceType1);
        ModelContext modelContext4 = new ModelContext(modelId1, knnEngine2, spaceType1);
        ModelContext modelContext5 = new ModelContext(modelId1, knnEngine1, spaceType2);

        assertEquals(modelContext1.hashCode(), modelContext1.hashCode());
        assertEquals(modelContext1.hashCode(), modelContext2.hashCode());
        assertNotEquals(modelContext1.hashCode(), modelContext3.hashCode());
        assertNotEquals(modelContext1.hashCode(), modelContext4.hashCode());
        assertNotEquals(modelContext1.hashCode(), modelContext5.hashCode());
    }

    public void testParse() {
        // Invalid input value
        Object invalidInput = 15;
        expectThrows(MapperParsingException.class, () -> ModelContext.parse(invalidInput));

        // Missing parameter
        expectThrows(MapperParsingException.class, () -> ModelContext.parse(ImmutableMap.of(
                KNN_ENGINE, KNNEngine.DEFAULT.getName(),
                METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue())));

        expectThrows(MapperParsingException.class, () -> ModelContext.parse(ImmutableMap.of(
                MODEL_ID, "test",
                METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue())));

        expectThrows(MapperParsingException.class, () -> ModelContext.parse(ImmutableMap.of(
                MODEL_ID, "test",
                KNN_ENGINE, KNNEngine.DEFAULT.getName())));

        // Extra parameter
        expectThrows(MapperParsingException.class, () -> ModelContext.parse(ImmutableMap.of(
                MODEL_ID, "test",
                KNN_ENGINE, KNNEngine.DEFAULT.getName(),
                METHOD_PARAMETER_SPACE_TYPE, SpaceType.DEFAULT.getValue(),
                "invalid", "invalid")));

        String modelId = "test-model";
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.DEFAULT;

        ModelContext modelContext = ModelContext.parse(ImmutableMap.of(
                MODEL_ID, modelId,
                KNN_ENGINE, knnEngine.getName(),
                METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue()));

        assertEquals(modelId, modelContext.getModelId());
        assertEquals(knnEngine, modelContext.getKNNEngine());
        assertEquals(spaceType, modelContext.getSpaceType());
    }
}
