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

import org.mockito.ArgumentCaptor;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FloatTrainingDataConsumerTests extends KNNTestCase {

    public void testAccept() {

        // Mock the training data allocation
        int dimension = 128;
        NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation = mock(NativeMemoryAllocation.TrainingDataAllocation.class); // new
                                                                                                                                          // NativeMemoryAllocation.TrainingDataAllocation(0,
                                                                                                                                          // numVectors*dimension*
                                                                                                                                          // Float.BYTES);
        when(trainingDataAllocation.getMemoryAddress()).thenReturn(0L);

        // Capture argument passed to set pointer
        ArgumentCaptor<Long> valueCapture = ArgumentCaptor.forClass(Long.class);

        FloatTrainingDataConsumer floatTrainingDataConsumer = new FloatTrainingDataConsumer(trainingDataAllocation);

        List<Float[]> vectorSet1 = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            Float[] vector = new Float[dimension];
            Arrays.fill(vector, (float) i);
            vectorSet1.add(vector);
        }

        // Transfer vectors
        floatTrainingDataConsumer.accept(vectorSet1);

        // Ensure that the pointer captured has been updated
        verify(trainingDataAllocation).setMemoryAddress(valueCapture.capture());
        when(trainingDataAllocation.getMemoryAddress()).thenReturn(valueCapture.getValue());

        assertNotEquals(0, trainingDataAllocation.getMemoryAddress());
    }
}
