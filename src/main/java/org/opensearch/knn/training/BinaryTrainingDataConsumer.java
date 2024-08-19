/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.training;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.search.SearchHit;

import java.util.ArrayList;
import java.util.List;

/**
 * Transfers binary vectors from JVM to native memory.
 */
public class BinaryTrainingDataConsumer extends TrainingDataConsumer {
    private static final Logger logger = LogManager.getLogger(TrainingDataConsumer.class);

    /**
     * Constructor
     *
     * @param trainingDataAllocation NativeMemoryAllocation that contains information about native memory allocation.
     */
    public BinaryTrainingDataConsumer(NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation) {
        super(trainingDataAllocation);
    }

    @Override
    public void accept(List<?> byteVectors) {
        long memoryAddress = trainingDataAllocation.getMemoryAddress();
        memoryAddress = JNICommons.storeBinaryVectorData(memoryAddress, byteVectors.toArray(new byte[0][0]), byteVectors.size());
        trainingDataAllocation.setMemoryAddress(memoryAddress);
    }

    @Override
    public void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName) {
        SearchHit[] hits = searchResponse.getHits().getHits();
        List<byte[]> vectors = new ArrayList<>();
        String[] fieldPath = fieldName.split("\\.");
        int nullVectorCount = 0;

        for (int vector = 0; vector < vectorsToAdd; vector++) {
            Object fieldValue = extractFieldValue(hits[vector], fieldPath);
            if (fieldValue == null) {
                nullVectorCount++;
                continue;
            }

            byte[] byteArray;
            if (!(fieldValue instanceof List<?>)) {
                continue;
            }
            List<Number> fieldList = (List<Number>) fieldValue;
            byteArray = new byte[fieldList.size()];
            for (int i = 0; i < fieldList.size(); i++) {
                byteArray[i] = fieldList.get(i).byteValue();
            }

            vectors.add(byteArray);
        }

        if (nullVectorCount > 0) {
            logger.warn("Found {} documents with null byte vectors in field {}", nullVectorCount, fieldName);
        }

        setTotalVectorsCountAdded(getTotalVectorsCountAdded() + vectors.size());

        accept(vectors);
    }
}
