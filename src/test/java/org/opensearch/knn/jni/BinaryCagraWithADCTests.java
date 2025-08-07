/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.apache.lucene.store.IndexInput;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import static org.opensearch.knn.common.KNNConstants.ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER;
import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class BinaryCagraWithADCTests extends KNNTestCase {
    @Test
    public void testBinaryCagraWithADC() {
        String indexPath = "data/memoryoptsearch/faiss_cagra2_flat_binary_300_vectors_768_dims.bin";
        final IndexInput binaryCagra = loadHnswBinary(indexPath);
        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(binaryCagra);
        final Map<String, Object> params = Map.of(
            ADC_ENABLED_FAISS_INDEX_INTERNAL_PARAMETER,
            true,
            KNNConstants.QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER,
            "ScalarQuantizationParams_1",
            KNNConstants.SPACE_TYPE_FAISS_INDEX_LOAD_PARAMETER,
            SpaceType.INNER_PRODUCT.getValue()
        );

        long indexPointer = JNIService.loadIndex(indexInputWithBuffer, params, KNNEngine.FAISS);
        try {
            int dimension = 768;
            float[] queryVector = new float[dimension];
            for (int i = 0; i < dimension; ++i) {
                queryVector[i] = ThreadLocalRandom.current().nextFloat();
            }
            int k = 10;

            final KNNQueryResult[] results = JNIService.queryIndex(
                indexPointer,
                queryVector,
                k,
                Collections.emptyMap(),
                KNNEngine.FAISS,
                null,
                0,
                null
            );

            assertEquals(k, results.length);
        } finally {
            JNIService.free(indexPointer, KNNEngine.FAISS);
        }
    }
}
