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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;

public class ModelInfoTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        ModelInfo modelInfo = new ModelInfo(knnEngine, spaceType, dimension);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        modelInfo.writeTo(streamOutput);

        ModelInfo modelInfoCopy = new ModelInfo(streamOutput.bytes().streamInput());

        assertEquals(modelInfo, modelInfoCopy);
    }

    public void testGetKnnEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelInfo modelInfo = new ModelInfo(knnEngine, SpaceType.L2, 128);

        assertEquals(knnEngine, modelInfo.getKnnEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.L2;
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, spaceType, 128);

        assertEquals(spaceType, modelInfo.getSpaceType());
    }

    public void testGetDimension() {
        int dimension = 128;
        ModelInfo modelInfo = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, dimension);

        assertEquals(dimension, modelInfo.getDimension());
    }

    public void testToString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String expected = knnEngine.getName() + "," + spaceType.getValue() + "," + dimension;

        ModelInfo modelInfo = new ModelInfo(knnEngine, spaceType, dimension);

        assertEquals(expected, modelInfo.toString());
    }

    public void testEquals() {
        ModelInfo modelInfo1 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelInfo modelInfo2 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelInfo modelInfo3 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 129);

        assertEquals(modelInfo1, modelInfo1);
        assertEquals(modelInfo1, modelInfo2);
        assertNotEquals(modelInfo1, null);
        assertNotEquals(modelInfo1, modelInfo3);
    }

    public void testHashCode() {
        ModelInfo modelInfo1 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelInfo modelInfo2 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelInfo modelInfo3 = new ModelInfo(KNNEngine.DEFAULT, SpaceType.L2, 129);

        assertEquals(modelInfo1.hashCode(), modelInfo1.hashCode());
        assertEquals(modelInfo1.hashCode(), modelInfo2.hashCode());
        assertNotEquals(modelInfo1.hashCode(), modelInfo3.hashCode());
    }

    public void testFromString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String stringRep1 = knnEngine.getName() + "," + spaceType.getValue() + "," + dimension;

        ModelInfo expected = new ModelInfo(knnEngine, spaceType, dimension);
        ModelInfo fromString1 = ModelInfo.fromString(stringRep1);

        assertEquals(expected, fromString1);

        expectThrows(IllegalArgumentException.class, () -> ModelInfo.fromString("invalid"));
    }
}
