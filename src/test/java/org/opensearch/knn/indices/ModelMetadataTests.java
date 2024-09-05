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

import org.opensearch.Version;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ModelMetadataTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        modelMetadata.writeTo(streamOutput);
        ModelMetadata modelMetadataCopy = new ModelMetadata(streamOutput.bytes().streamInput());
        assertEquals(modelMetadata, modelMetadataCopy);

        modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.ON_DISK,
            CompressionLevel.x16,
            Version.CURRENT
        );
        streamOutput = new BytesStreamOutput();
        modelMetadata.writeTo(streamOutput);
        modelMetadataCopy = new ModelMetadata(streamOutput.bytes().streamInput());
        assertEquals(modelMetadata, modelMetadataCopy);
    }

    public void testGetKnnEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(knnEngine, modelMetadata.getKnnEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.L2;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            spaceType,
            128,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(spaceType, modelMetadata.getSpaceType());
    }

    public void testGetDimension() {
        int dimension = 128;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(dimension, modelMetadata.getDimension());
    }

    public void testGetState() {
        ModelState modelState = ModelState.FAILED;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            modelState,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(modelState, modelMetadata.getState());
    }

    public void testGetTimestamp() {
        String timeValue = ZonedDateTime.now(ZoneOffset.UTC).toString();
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.CREATED,
            timeValue,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(timeValue, modelMetadata.getTimestamp());
    }

    public void testDescription() {
        String description = "test description";
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            description,
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(description, modelMetadata.getDescription());
    }

    public void testGetError() {
        String error = "test error";
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            error,
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(error, modelMetadata.getError());
    }

    public void testGetVectorDataType() {
        VectorDataType vectorDataType = VectorDataType.BINARY;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            vectorDataType,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(vectorDataType, modelMetadata.getVectorDataType());
    }

    public void testGetModelVersion() {
        Version version = Version.CURRENT;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            version
        );

        assertEquals(version, modelMetadata.getModelVersion());
    }

    public void testSetState() {
        ModelState modelState = ModelState.FAILED;
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            modelState,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(modelState, modelMetadata.getState());

        ModelState updatedState = ModelState.CREATED;
        modelMetadata.setState(updatedState);
        assertEquals(updatedState, modelMetadata.getState());
    }

    public void testSetError() {
        String error = "";
        ModelMetadata modelMetadata = new ModelMetadata(
            KNNEngine.DEFAULT,
            SpaceType.L2,
            12,
            ModelState.TRAINING,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            error,
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

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
        String nodeAssignment = "";
        MethodComponentContext methodComponentContext = MethodComponentContext.EMPTY;
        Version version = Version.CURRENT;

        String expected = knnEngine.getName()
            + ","
            + spaceType.getValue()
            + ","
            + dimension
            + ","
            + modelState.getName()
            + ","
            + timestamp
            + ","
            + description
            + ","
            + error
            + ","
            + nodeAssignment
            + ","
            + methodComponentContext.toClusterStateString()
            + ","
            + VectorDataType.DEFAULT.getValue()
            + ","
            + Mode.ON_DISK.getName()
            + ","
            + CompressionLevel.x32.getName()
            + ","
            + version;

        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            nodeAssignment,
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.ON_DISK,
            CompressionLevel.x32,
            Version.CURRENT
        );

        assertEquals(expected, modelMetadata.toString());
    }

    public void testEquals() {

        String time1 = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String time2 = ZonedDateTime.of(2021, 9, 30, 12, 20, 45, 1, ZoneId.systemDefault()).toString();

        ModelMetadata modelMetadata1 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata2 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        ModelMetadata modelMetadata3 = new ModelMetadata(
            KNNEngine.NMSLIB,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata4 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L1,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata5 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            129,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata6 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.TRAINING,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata7 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time2,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata8 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "diff descript",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata9 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "diff error",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        ModelMetadata modelMetadata10 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            new MethodComponentContext("test", Collections.emptyMap()),
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

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
        String time2 = ZonedDateTime.of(2021, 9, 30, 12, 20, 45, 1, ZoneId.systemDefault()).toString();

        ModelMetadata modelMetadata1 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata2 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        ModelMetadata modelMetadata3 = new ModelMetadata(
            KNNEngine.NMSLIB,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata4 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L1,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata5 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            129,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata6 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.TRAINING,
            time1,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata7 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time2,
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata8 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "diff descript",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );
        ModelMetadata modelMetadata9 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "diff error",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        ModelMetadata modelMetadata10 = new ModelMetadata(
            KNNEngine.FAISS,
            SpaceType.L2,
            128,
            ModelState.CREATED,
            time1,
            "",
            "",
            "",
            new MethodComponentContext("test", Collections.emptyMap()),
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        assertEquals(modelMetadata1.hashCode(), modelMetadata1.hashCode());
        assertEquals(modelMetadata1.hashCode(), modelMetadata2.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata3.hashCode());

        assertNotEquals(modelMetadata1.hashCode(), modelMetadata4.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata5.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata6.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata7.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata8.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata9.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata10.hashCode());
    }

    public void testFromString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";
        String nodeAssignment = "test-node";
        MethodComponentContext methodComponentContext = MethodComponentContext.EMPTY;
        Version version = Version.CURRENT;

        String stringRep1 = knnEngine.getName()
            + ","
            + spaceType.getValue()
            + ","
            + dimension
            + ","
            + modelState.getName()
            + ","
            + timestamp
            + ","
            + description
            + ","
            + error
            + ","
            + nodeAssignment
            + ","
            + methodComponentContext.toClusterStateString()
            + ","
            + VectorDataType.DEFAULT.getValue()
            + ","
            + ","
            + ","
            + version.toString();

        String stringRep2 = knnEngine.getName()
            + ","
            + spaceType.getValue()
            + ","
            + dimension
            + ","
            + modelState.getName()
            + ","
            + timestamp
            + ","
            + description
            + ","
            + error
            + ","
            + VectorDataType.DEFAULT.getValue();

        String stringRep3 = knnEngine.getName()
            + ","
            + spaceType.getValue()
            + ","
            + dimension
            + ","
            + modelState.getName()
            + ","
            + timestamp
            + ","
            + description
            + ","
            + error
            + ","
            + nodeAssignment
            + ","
            + methodComponentContext.toClusterStateString()
            + ","
            + VectorDataType.DEFAULT.getValue()
            + ","
            + ","
            + ","
            + version;

        String stringRep4 = knnEngine.getName()
            + ","
            + spaceType.getValue()
            + ","
            + dimension
            + ","
            + modelState.getName()
            + ","
            + timestamp
            + ","
            + description
            + ","
            + error
            + ","
            + nodeAssignment
            + ","
            + methodComponentContext.toClusterStateString()
            + ","
            + VectorDataType.DEFAULT.getValue()
            + ","
            + Mode.ON_DISK.getName()
            + ","
            + CompressionLevel.x32.getName();

        ModelMetadata expected1 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            nodeAssignment,
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.V_EMPTY
        );

        ModelMetadata expected2 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            Version.V_EMPTY
        );

        ModelMetadata expected3 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT,
            Mode.ON_DISK,
            CompressionLevel.x32,
            Version.CURRENT
        );

        ModelMetadata fromString1 = ModelMetadata.fromString(stringRep1);
        ModelMetadata fromString2 = ModelMetadata.fromString(stringRep2);
        ModelMetadata fromString3 = ModelMetadata.fromString(stringRep3);
        ModelMetadata fromString4 = ModelMetadata.fromString(stringRep4);

        assertEquals(expected1, fromString1);
        assertEquals(expected2, fromString2);
        assertEquals(expected2, fromString3);
        assertEquals(expected3, fromString4);

        expectThrows(IllegalArgumentException.class, () -> ModelMetadata.fromString("invalid"));
    }

    public void testFromResponseMap() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";
        String nodeAssignment = "test-node";
        MethodComponentContext methodComponentContext = getMethodComponentContext();
        MethodComponentContext emptyMethodComponentContext = MethodComponentContext.EMPTY;
        Version version = Version.CURRENT;
        Version emptyVersion = Version.V_EMPTY;

        ModelMetadata expected = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            nodeAssignment,
            methodComponentContext,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            version
        );
        ModelMetadata expected2 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            "",
            emptyMethodComponentContext,
            VectorDataType.DEFAULT,
            Mode.NOT_CONFIGURED,
            CompressionLevel.NOT_CONFIGURED,
            emptyVersion
        );

        ModelMetadata expected3 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            "",
            emptyMethodComponentContext,
            VectorDataType.DEFAULT,
            Mode.ON_DISK,
            CompressionLevel.NOT_CONFIGURED,
            Version.CURRENT
        );

        ModelMetadata expected4 = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            "",
            emptyMethodComponentContext,
            VectorDataType.DEFAULT,
            Mode.ON_DISK,
            CompressionLevel.x16,
            Version.CURRENT
        );
        Map<String, Object> metadataAsMap = new HashMap<>();
        metadataAsMap.put(KNNConstants.KNN_ENGINE, knnEngine.getName());
        metadataAsMap.put(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        metadataAsMap.put(KNNConstants.DIMENSION, dimension);
        metadataAsMap.put(KNNConstants.MODEL_STATE, modelState.getName());
        metadataAsMap.put(KNNConstants.MODEL_TIMESTAMP, timestamp);
        metadataAsMap.put(KNNConstants.MODEL_DESCRIPTION, description);
        metadataAsMap.put(KNNConstants.MODEL_ERROR, error);
        metadataAsMap.put(KNNConstants.MODEL_NODE_ASSIGNMENT, nodeAssignment);
        metadataAsMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue());

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        builder = methodComponentContext.toXContent(builder, ToXContent.EMPTY_PARAMS).endObject();
        metadataAsMap.put(KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT, builder.toString());
        metadataAsMap.put(KNNConstants.MODEL_VERSION, version.toString());

        ModelMetadata fromMap = ModelMetadata.getMetadataFromSourceMap(metadataAsMap);
        assertEquals(expected, fromMap);

        metadataAsMap.put(KNNConstants.MODEL_NODE_ASSIGNMENT, null);
        metadataAsMap.put(KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT, null);
        metadataAsMap.put(KNNConstants.MODEL_VERSION, emptyVersion);
        assertEquals(expected2, fromMap);

        metadataAsMap.put(KNNConstants.MODE_PARAMETER, Mode.ON_DISK.getName());
        fromMap = ModelMetadata.getMetadataFromSourceMap(metadataAsMap);
        assertEquals(expected3, fromMap);

        metadataAsMap.put(KNNConstants.COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName());
        fromMap = ModelMetadata.getMetadataFromSourceMap(metadataAsMap);
        assertEquals(expected4, fromMap);
    }

    public void testBlockCommasInDescription() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.TRAINING;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "Test, comma, description";
        String error = "test-error";
        String nodeAssignment = "test-node";
        MethodComponentContext methodComponentContext = getMethodComponentContext();

        Exception e = expectThrows(
            IllegalArgumentException.class,
            () -> new ModelMetadata(
                knnEngine,
                spaceType,
                dimension,
                modelState,
                timestamp,
                description,
                error,
                nodeAssignment,
                methodComponentContext,
                VectorDataType.DEFAULT,
                Mode.NOT_CONFIGURED,
                CompressionLevel.NOT_CONFIGURED,
                Version.CURRENT
            )
        );
        assertEquals("Model description cannot contain any commas: ','", e.getMessage());
    }

    private static MethodComponentContext getMethodComponentContext() {
        Map<String, Object> nestedParameters = new HashMap<String, Object>() {
            {
                put("testNestedKey1", "testNestedString");
                put("testNestedKey2", 1);
            }
        };
        Map<String, Object> parameters = new HashMap<>() {
            {
                put("testKey1", "testString");
                put("testKey2", 0);
                put("testKey3", new MethodComponentContext("ivf", nestedParameters));
            }
        };
        MethodComponentContext methodComponentContext = new MethodComponentContext("hnsw", parameters);
        return methodComponentContext;
    }
}
