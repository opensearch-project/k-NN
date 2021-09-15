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

import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import static org.opensearch.knn.index.KNNVectorFieldMapper.MAX_DIMENSION;

public class ModelTests extends KNNTestCase {

    public void testNullConstructor() {
        expectThrows(NullPointerException.class, () -> new Model(null, null));
    }

    public void testInvalidConstructor() {
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelMetadata(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, -1, ModelState.FAILED, TimeValue.timeValueDays(10), "", ""),
                null));
    }

    public void testInvalidDimension() {
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelMetadata(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, -1, ModelState.CREATED, TimeValue.timeValueDays(10), "", ""),
                new byte[16]));
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelMetadata(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, 0, ModelState.CREATED, TimeValue.timeValueDays(10), "", ""),
                new byte[16]));
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelMetadata(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, MAX_DIMENSION + 1, ModelState.CREATED, TimeValue.timeValueDays(10), "", ""),
                new byte[16]));
    }

    public void testGetModelMetadata() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, SpaceType.DEFAULT, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", "");
        Model model = new Model(modelMetadata, new byte[16]);
        assertEquals(modelMetadata, model.getModelMetadata());
    }

    public void testGetModelBlob() {
        byte[] modelBlob = "hello".getBytes();
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), modelBlob);
        assertArrayEquals(modelBlob, model.getModelBlob());
    }

    public void testGetLength() {
        int size = 129;
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[size]);
        assertEquals(size, model.getLength());

        model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2, ModelState.TRAINING,
                TimeValue.timeValueDays(10), "", ""), null);
        assertEquals(0, model.getLength());
    }

    public void testSetModelBlob() {
        byte[] blob1 = "Hello blob 1".getBytes();
        Model model = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), blob1);
        assertEquals(blob1, model.getModelBlob());
        byte[] blob2 = "Hello blob 2".getBytes();

        model.setModelBlob(blob2);
        assertEquals(blob2, model.getModelBlob());
    }

    public void testEquals() {
        Model model1 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);
        Model model2 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);
        Model model3 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);
        Model model4 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[32]);
        Model model5 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 4, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);

        assertEquals(model1, model1);
        assertEquals(model1, model2);
        assertNotEquals(model1, model3);
        assertNotEquals(model1, model4);
        assertNotEquals(model1, model5);
    }

    public void testHashCode() {
        Model model1 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);
        Model model2 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);
        Model model3 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L1, 2, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[32]);
        Model model4 = new Model(new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 4, ModelState.CREATED,
                TimeValue.timeValueDays(10), "", ""), new byte[16]);

        assertEquals(model1.hashCode(), model1.hashCode());
        assertEquals(model1.hashCode(), model2.hashCode());
        assertNotEquals(model1.hashCode(), model3.hashCode());
        assertNotEquals(model1.hashCode(), model4.hashCode());
    }
}
