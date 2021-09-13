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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import static org.opensearch.knn.index.KNNVectorFieldMapper.MAX_DIMENSION;

public class ModelTests extends KNNTestCase {

    public void testNullConstructor() {
        expectThrows(NullPointerException.class, () -> new Model(null, null));
    }

    public void testInvalidDimension() {
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelInfo(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, -1), new byte[16]));
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelInfo(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, 0), new byte[16]));
        expectThrows(IllegalArgumentException.class, () -> new Model(new ModelInfo(KNNEngine.DEFAULT,
                SpaceType.DEFAULT, MAX_DIMENSION + 1), new byte[16]));
    }

    public void testGetModelInfo() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelInfo modelInfo = new ModelInfo(knnEngine, SpaceType.DEFAULT, 2);
        Model model = new Model(modelInfo, new byte[16]);
        assertEquals(modelInfo, model.getModelInfo());
    }

    public void testGetModelBlob() {
        byte[] modelBlob = "hello".getBytes();
        Model model = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2), modelBlob);
        assertArrayEquals(modelBlob, model.getModelBlob());
    }

    public void testGetLength() {
        int size = 129;
        Model model = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.DEFAULT, 2), new byte[size]);
        assertEquals(size, model.getLength());
    }

    public void testEquals() {
        Model model1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[16]);
        Model model2 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[16]);
        Model model3 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 2), new byte[16]);
        Model model4 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[32]);
        Model model5 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 4), new byte[16]);

        assertEquals(model1, model1);
        assertEquals(model1, model2);
        assertNotEquals(model1, model3);
        assertNotEquals(model1, model4);
        assertNotEquals(model1, model5);
    }

    public void testHashCode() {
        Model model1 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[16]);
        Model model2 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[16]);
        Model model3 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L1, 2), new byte[32]);
        Model model4 = new Model(new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 4), new byte[16]);

        assertEquals(model1.hashCode(), model1.hashCode());
        assertEquals(model1.hashCode(), model2.hashCode());
        assertNotEquals(model1.hashCode(), model3.hashCode());
        assertNotEquals(model1.hashCode(), model4.hashCode());
    }
}
