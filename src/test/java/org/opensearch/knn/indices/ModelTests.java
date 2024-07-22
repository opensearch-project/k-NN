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

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.Map;

public class ModelTests extends KNNTestCase {

    public void testNullConstructor() {
        expectThrows(NullPointerException.class, () -> new Model(null, null, "test-model"));
    }

    public void testInvalidConstructor() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    -1,
                    ModelState.FAILED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    "",
                    "",
                    MethodComponentContext.EMPTY,
                    VectorDataType.DEFAULT
                ),
                null,
                "test-model"
            )
        );
    }

    public void testInvalidDimension() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    -1,
                    ModelState.CREATED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    "",
                    "",
                    MethodComponentContext.EMPTY,
                    VectorDataType.DEFAULT
                ),
                new byte[16],
                "test-model"
            )
        );
        expectThrows(
            IllegalArgumentException.class,
            () -> new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    0,
                    ModelState.CREATED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    "",
                    "",
                    MethodComponentContext.EMPTY,
                    VectorDataType.DEFAULT
                ),
                new byte[16],
                "test-model"
            )
        );
        expectThrows(
            IllegalArgumentException.class,
            () -> new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    KNNEngine.getMaxDimensionByEngine(KNNEngine.DEFAULT) + 1,
                    ModelState.CREATED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    "",
                    "",
                    MethodComponentContext.EMPTY,
                    VectorDataType.DEFAULT
                ),
                new byte[16],
                "test-model"
            )
        );
    }

    public void testGetModelMetadata() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            SpaceType.DEFAULT,
            2,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "",
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );
        Model model = new Model(modelMetadata, new byte[16], "test-model");
        assertEquals(modelMetadata, model.getModelMetadata());
    }

    public void testGetModelBlob() {
        byte[] modelBlob = "hello".getBytes();
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                2,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            modelBlob,
            "test-model"
        );
        assertArrayEquals(modelBlob, model.getModelBlob());
    }

    public void testGetLength() {
        int size = 129;
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                2,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[size],
            "test-model"
        );
        assertEquals(size, model.getLength());

        model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.DEFAULT,
                2,
                ModelState.TRAINING,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            null,
            "test-model"
        );
        assertEquals(0, model.getLength());
    }

    public void testSetModelBlob() {
        byte[] blob1 = "Hello blob 1".getBytes();
        Model model = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                ZonedDateTime.now(ZoneOffset.UTC).toString(),
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            blob1,
            "test-model"
        );
        assertEquals(blob1, model.getModelBlob());
        byte[] blob2 = "Hello blob 2".getBytes();

        model.setModelBlob(blob2);
        assertEquals(blob2, model.getModelBlob());
    }

    public void testEquals() {

        String time = ZonedDateTime.now(ZoneOffset.UTC).toString();

        Model model1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-1"
        );
        Model model2 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-1"
        );
        Model model3 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L2,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-2"
        );

        assertEquals(model1, model1);
        assertEquals(model1, model2);
        assertNotEquals(model1, model3);
    }

    public void testHashCode() {

        String time = ZonedDateTime.now(ZoneOffset.UTC).toString();

        Model model1 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-1"
        );
        Model model2 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-1"
        );
        Model model3 = new Model(
            new ModelMetadata(
                KNNEngine.DEFAULT,
                SpaceType.L1,
                2,
                ModelState.CREATED,
                time,
                "",
                "",
                "",
                MethodComponentContext.EMPTY,
                VectorDataType.DEFAULT
            ),
            new byte[16],
            "test-model-2"
        );

        assertEquals(model1.hashCode(), model1.hashCode());
        assertEquals(model1.hashCode(), model2.hashCode());
        assertNotEquals(model1.hashCode(), model3.hashCode());
    }

    public void testModelFromSourceMap() {
        String modelID = "test-modelid";
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;
        ModelState modelState = ModelState.CREATED;
        String timestamp = ZonedDateTime.now(ZoneOffset.UTC).toString();
        String description = "test-description";
        String error = "test-error";
        String nodeAssignment = "test-node";

        ModelMetadata metadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            nodeAssignment,
            MethodComponentContext.EMPTY,
            VectorDataType.DEFAULT
        );
        Map<String, Object> modelAsMap = new HashMap<>();
        modelAsMap.put(KNNConstants.MODEL_ID, modelID);
        modelAsMap.put(KNNConstants.KNN_ENGINE, knnEngine.getName());
        modelAsMap.put(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue());
        modelAsMap.put(KNNConstants.DIMENSION, dimension);
        modelAsMap.put(KNNConstants.MODEL_STATE, modelState.getName());
        modelAsMap.put(KNNConstants.MODEL_TIMESTAMP, timestamp);
        modelAsMap.put(KNNConstants.MODEL_DESCRIPTION, description);
        modelAsMap.put(KNNConstants.MODEL_ERROR, error);
        modelAsMap.put(KNNConstants.MODEL_NODE_ASSIGNMENT, nodeAssignment);
        modelAsMap.put(KNNConstants.MODEL_BLOB_PARAMETER, "aGVsbG8=");
        modelAsMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue());

        byte[] blob1 = "hello".getBytes();
        Model expected = new Model(metadata, blob1, modelID);

        Model fromMap = Model.getModelFromSourceMap(modelAsMap);
        assertEquals(expected, fromMap);
    }
}
