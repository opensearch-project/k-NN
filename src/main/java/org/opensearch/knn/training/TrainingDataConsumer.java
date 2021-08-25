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

import org.apache.commons.lang.ArrayUtils;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;

import java.util.List;
import java.util.function.Consumer;

/**
 * Transfers vectors from JVM to native memory.
 */
public class TrainingDataConsumer implements Consumer<List<Float[]>> {

    private final NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation;

    /**
     * Constructor
     *
     * @param trainingDataAllocation NativeMemoryAllocation that contains information about native memory allocation.
     */
    public TrainingDataConsumer(NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation) {
        this.trainingDataAllocation = trainingDataAllocation;
    }

    @Override
    public void accept(List<Float[]> floats) {
        trainingDataAllocation.setMemoryAddress(JNIService.transferVectors(trainingDataAllocation.getMemoryAddress(),
                floats.stream().map(ArrayUtils::toPrimitive).toArray(float[][]::new)));
    }
}
