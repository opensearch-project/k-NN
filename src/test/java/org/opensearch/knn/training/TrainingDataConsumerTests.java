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

package org.opensearch.knn.training;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainingDataConsumerTests extends KNNTestCase {

    public void testAccept() {
        int numVectors = 10;
        int dimension = 128;

        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = new NativeMemoryAllocation.TrainingDataAllocation(0, numVectors*dimension* Float.BYTES);
        TrainingDataConsumer trainingDataConsumer = new TrainingDataConsumer(trainingDataAllocation);
        assertEquals(0, trainingDataAllocation.getPointer());

        List<Float[]> vectorSet1 = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            Float[] vector = new Float[dimension];
            Arrays.fill(vector, (float) i);
            vectorSet1.add(vector);
        }

        trainingDataConsumer.accept(vectorSet1);
        long pointer = trainingDataAllocation.getPointer();
        assertNotEquals(0, pointer);

        List<Float[]> vectorSet2 = new ArrayList<>(3);
        for (int i = 0; i < 7; i++) {
            Float[] vector = new Float[dimension];
            Arrays.fill(vector, (float) i);
            vectorSet2.add(vector);
        }

        trainingDataConsumer.accept(vectorSet2);
        assertEquals(pointer, pointer);
    }
}
