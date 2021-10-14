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

package org.opensearch.knn.indices;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.Map;

public class ModelMetadataTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, spaceType, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        modelMetadata.writeTo(streamOutput);

        ModelMetadata modelMetadataCopy = new ModelMetadata(streamOutput.bytes().streamInput());

        assertEquals(modelMetadata, modelMetadataCopy);
    }

    public void testGetKnnEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, SpaceType.L2, 128, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        assertEquals(knnEngine, modelMetadata.getKnnEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.L2;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, spaceType, 128, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        assertEquals(spaceType, modelMetadata.getSpaceType());
    }

    public void testGetDimension() {
        int dimension = 128;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, dimension, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        assertEquals(dimension, modelMetadata.getDimension());
    }

    public void testGetState() {
        ModelState modelState = ModelState.FAILED;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, modelState,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        assertEquals(modelState, modelMetadata.getState());
    }

    public void testGetTimestamp() {
        String timeValue = ZonedDateTime.now(ZoneOffset.UTC).toString();
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, ModelState.CREATED,
                timeValue, "", "");

        assertEquals(timeValue, modelMetadata.getTimestamp());
    }

    public void testDescription() {
        String description = "test description";
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), description, "");

        assertEquals(description, modelMetadata.getDescription());
    }

    public void testGetError() {
        String error = "test error";
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", error);

        assertEquals(error, modelMetadata.getError());
    }

    public void testSetState() {
        ModelState modelState = ModelState.FAILED;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, modelState,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", "");

        assertEquals(modelState, modelMetadata.getState());

        ModelState updatedState = ModelState.CREATED;
        modelMetadata.setState(updatedState);
        assertEquals(updatedState, modelMetadata.getState());
    }

    public void testSetError() {
        String error = "";
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 12, ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(), "", error);

        assertEquals(error, modelMetadata.getError());

        String updatedError = "test error";
        modelMetadata.setError(updatedError);
        assertEquals(updatedError, modelMetadata.getError());
    }

    public void testToString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";

        String expected = knnEngine.getName() + "," +
                spaceType.getValue() + "," +
                dimension + "," +
                modelState.getName() + "," +
                timestamp + "," +
                description + "," +
                error;

        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, spaceType, dimension, modelState,
                timestamp, description, error);

        assertEquals(expected, modelMetadata.toString());
    }

    public void testEquals() {

        String time1 = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String time2 = ZonedDateTime.of(2021, 9, 30,12, 20, 45, 1,
                ZoneId.systemDefault()).toString();

        ModelMetadata modelMetadata1 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata2 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");

        ModelMetadata modelMetadata3 = new ModelMetadata(KNNEngine.NMSLIB, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata4 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L1, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata5 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 129, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata6 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.TRAINING,
                time1, "", "");
        ModelMetadata modelMetadata7 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time2, "", "");
        ModelMetadata modelMetadata8 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "diff descript", "");
        ModelMetadata modelMetadata9 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "diff error");

        assertEquals(modelMetadata1, modelMetadata1);
        assertEquals(modelMetadata1, modelMetadata2);
        assertNotEquals(modelMetadata1, null);

        assertNotEquals(modelMetadata1, modelMetadata3);
        assertNotEquals(modelMetadata1, modelMetadata4);
        assertNotEquals(modelMetadata1, modelMetadata5);
        assertNotEquals(modelMetadata1, modelMetadata6);
        assertNotEquals(modelMetadata1, modelMetadata7);
        assertNotEquals(modelMetadata1, modelMetadata8);
        assertNotEquals(modelMetadata1, modelMetadata9);
    }

    public void testHashCode() {

        String time1 = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String time2 = ZonedDateTime.of(2021, 9, 30,12, 20, 45, 1,
                ZoneId.systemDefault()).toString();

        ModelMetadata modelMetadata1 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata2 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");

        ModelMetadata modelMetadata3 = new ModelMetadata(KNNEngine.NMSLIB, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata4 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L1, 128, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata5 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 129, ModelState.CREATED,
                time1, "", "");
        ModelMetadata modelMetadata6 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.TRAINING,
                time1, "", "");
        ModelMetadata modelMetadata7 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time2, "", "");
        ModelMetadata modelMetadata8 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "diff descript", "");
        ModelMetadata modelMetadata9 = new ModelMetadata(KNNEngine.FAISS, SpaceType.L2, 128, ModelState.CREATED,
                time1, "", "diff error");

        assertEquals(modelMetadata1.hashCode(), modelMetadata1.hashCode());
        assertEquals(modelMetadata1.hashCode(), modelMetadata2.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata3.hashCode());

        assertNotEquals(modelMetadata1.hashCode(), modelMetadata3.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata4.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata5.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata6.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata7.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata8.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata9.hashCode());
    }

    public void testFromString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";

        String stringRep1 = knnEngine.getName() + "," +
                spaceType.getValue() + "," +
                dimension + "," +
                modelState.getName() + "," +
                timestamp + "," +
                description + "," +
                error;


        ModelMetadata expected = new ModelMetadata(knnEngine, spaceType, dimension, modelState,
                timestamp, description, error);
        ModelMetadata fromString1 = ModelMetadata.fromString(stringRep1);

        assertEquals(expected, fromString1);

        expectThrows(IllegalArgumentException.class, () -> ModelMetadata.fromString("invalid"));
    }

    public void testFromResponseMap() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";

        ModelMetadata expected = new ModelMetadata(knnEngine, spaceType, dimension, modelState,
            timestamp, description, error);
        Map<String,Object> metadataAsMap = new HashMap<>();
        metadataAsMap.put(KNNConstants.KNN_ENGINE, knnEngine.getName());
        metadataAsMap.put(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        metadataAsMap.put(KNNConstants.DIMENSION, dimension);
        metadataAsMap.put(KNNConstants.MODEL_STATE, modelState.getName());
        metadataAsMap.put(KNNConstants.MODEL_TIMESTAMP, timestamp);
        metadataAsMap.put(KNNConstants.MODEL_DESCRIPTION, description);
        metadataAsMap.put(KNNConstants.MODEL_ERROR, error);

        ModelMetadata fromMap = ModelMetadata.getMetadataFromSourceMap(metadataAsMap);
        assertEquals(expected, fromMap);
    }
}
