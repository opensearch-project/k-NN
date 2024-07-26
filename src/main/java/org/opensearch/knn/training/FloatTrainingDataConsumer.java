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
import org.opensearch.action.search.SearchResponse;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.search.SearchHit;

import java.util.ArrayList;
import java.util.List;

/**
 * Transfers float vectors from JVM to native memory.
 */
public class FloatTrainingDataConsumer extends TrainingDataConsumer {

    /**
     * Constructor
     *
     * @param trainingDataAllocation NativeMemoryAllocation that contains information about native memory allocation.
     */
    public FloatTrainingDataConsumer(NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation) {
        super(trainingDataAllocation);
    }

    @Override
    public void accept(List<?> floats) {
        trainingDataAllocation.setMemoryAddress(
            JNIService.transferVectors(
                trainingDataAllocation.getMemoryAddress(),
                floats.stream().map(v -> ArrayUtils.toPrimitive((Float[]) v)).toArray(float[][]::new)
            )
        );
    }

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<Float[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);
            if (!(fieldValue instanceof List<?>)) {
                continue;
            }

            List<Number> fieldList = (List<Number>) fieldValue;
            vectors.add(fieldList.stream().map(Number::floatValue).toArray(Float[]::new));
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }
}
