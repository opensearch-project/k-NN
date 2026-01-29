/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.util.Collections;
import java.util.concurrent.ThreadLocalRandom;

import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class BinaryCagraHammingScoreConversionTests extends KNNTestCase {
    @Test
    @SneakyThrows
    public void testBinaryCagraScoreConversion() {
        // Binary cagra with 300 vectors 32 dims, having `base_level_only = true`
        final String indexPath = "data/memoryoptsearch/faiss_binary_cagra_300_vectors_32_dims.faiss";
        final IndexInput binaryCagra = loadHnswBinary(indexPath);
        final IndexInputWithBuffer indexInputWithBuffer = new IndexInputWithBuffer(binaryCagra);

        // Load Faiss binary HNSW graph
        long indexPointer = FaissService.loadBinaryIndexWithStream(indexInputWithBuffer);

        // Parameters
        final int dimension = 4;
        final int k = 10;
        final KNNQueryResult[] results;
        final byte[] queryVector = new byte[dimension];
        try {

            // Make a query vector
            for (int i = 0; i < dimension; ++i) {
                queryVector[i] = (byte) ThreadLocalRandom.current().nextInt();
            }

            // Search
            results = JNIService.queryBinaryIndex(indexPointer, queryVector, k, Collections.emptyMap(), KNNEngine.FAISS, null, 0, null);

            assertEquals(k, results.length);
        } finally {
            JNIService.free(indexPointer, KNNEngine.FAISS);
        }

        // Validate score value
        binaryCagra.seek(0);
        final FaissIndex faissIndex = FaissIndex.load(binaryCagra);
        final IndexInput vectorReadInput = binaryCagra.clone();
        final ByteVectorValues byteValues = faissIndex.getByteValues(vectorReadInput);

        // Make sure hamming distance score conversion made properly.
        for (final KNNQueryResult result : results) {
            final byte[] quantizedVector = byteValues.vectorValue(result.getId());
            final float obtainedScore = SpaceType.HAMMING.scoreTranslation(result.getScore());
            final float expectedScore = SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(queryVector, quantizedVector);
            assertEquals(expectedScore, obtainedScore, 1e-3);
        }
    }
}
