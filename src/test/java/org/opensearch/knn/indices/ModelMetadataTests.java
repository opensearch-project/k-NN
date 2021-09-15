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

public class ModelMetadataTests extends KNNTestCase {

    public void testStreams() throws IOException {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, spaceType, dimension);

        BytesStreamOutput streamOutput = new BytesStreamOutput();
        modelMetadata.writeTo(streamOutput);

        ModelMetadata modelMetadataCopy = new ModelMetadata(streamOutput.bytes().streamInput());

        assertEquals(modelMetadata, modelMetadataCopy);
    }

    public void testGetKnnEngine() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, SpaceType.L2, 128);

        assertEquals(knnEngine, modelMetadata.getKnnEngine());
    }

    public void testGetSpaceType() {
        SpaceType spaceType = SpaceType.L2;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, spaceType, 128);

        assertEquals(spaceType, modelMetadata.getSpaceType());
    }

    public void testGetDimension() {
        int dimension = 128;
        ModelMetadata modelMetadata = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, dimension);

        assertEquals(dimension, modelMetadata.getDimension());
    }

    public void testToString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String expected = knnEngine.getName() + "," + spaceType.getValue() + "," + dimension;

        ModelMetadata modelMetadata = new ModelMetadata(knnEngine, spaceType, dimension);

        assertEquals(expected, modelMetadata.toString());
    }

    public void testEquals() {
        ModelMetadata modelMetadata1 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelMetadata modelMetadata2 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelMetadata modelMetadata3 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 129);

        assertEquals(modelMetadata1, modelMetadata1);
        assertEquals(modelMetadata1, modelMetadata2);
        assertNotEquals(modelMetadata1, null);
        assertNotEquals(modelMetadata1, modelMetadata3);
    }

    public void testHashCode() {
        ModelMetadata modelMetadata1 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelMetadata modelMetadata2 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 128);
        ModelMetadata modelMetadata3 = new ModelMetadata(KNNEngine.DEFAULT, SpaceType.L2, 129);

        assertEquals(modelMetadata1.hashCode(), modelMetadata1.hashCode());
        assertEquals(modelMetadata1.hashCode(), modelMetadata2.hashCode());
        assertNotEquals(modelMetadata1.hashCode(), modelMetadata3.hashCode());
    }

    public void testFromString() {
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 128;

        String stringRep1 = knnEngine.getName() + "," + spaceType.getValue() + "," + dimension;

        ModelMetadata expected = new ModelMetadata(knnEngine, spaceType, dimension);
        ModelMetadata fromString1 = ModelMetadata.fromString(stringRep1);

        assertEquals(expected, fromString1);

        expectThrows(IllegalArgumentException.class, () -> ModelMetadata.fromString("invalid"));
    }
}
