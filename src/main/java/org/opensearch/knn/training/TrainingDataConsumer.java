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

import lombok.Getter;
import lombok.Setter;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.search.SearchHit;

import java.util.List;
import java.util.Map;

/**
 * TrainingDataConsumer is an abstract class that defines the interface for consuming training data.
 * It is used to process training data and add it to the training data allocation.
 */
public abstract class TrainingDataConsumer {

    @Setter
    @Getter
    private int totalVectorsCountAdded = 0;
    protected final NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation;

    /**
     * Constructor
     *
     * @param trainingDataAllocation NativeMemoryAllocation that contains information about native memory allocation.
     */
    public TrainingDataConsumer(NativeMemoryAllocation.TrainingDataAllocation trainingDataAllocation) {
        this.trainingDataAllocation = trainingDataAllocation;
    }

    protected abstract void accept(List<?> vectors);

    public abstract void processTrainingVectors(SearchResponse searchResponse, int vectorsToAdd, String fieldName);

    /**
     * Traverses the hit to the desired field and extracts its value.
     *
     * @param hit The search hit to extract the field value from
     * @param fieldPath The path to the desired field
     * @return The extracted field value, or null if the field does not exist
     */
    protected Object extractFieldValue(SearchHit hit, String[] fieldPath) {
        Map<String, Object> currentMap = hit.getSourceAsMap();
        for (int pathPart = 0; pathPart < fieldPath.length - 1; pathPart++) {
            currentMap = (Map<String, Object>) currentMap.get(fieldPath[pathPart]);
            if (currentMap == null) {
                return null;
            }
        }
        return currentMap.get(fieldPath[fieldPath.length - 1]);
    }
}
