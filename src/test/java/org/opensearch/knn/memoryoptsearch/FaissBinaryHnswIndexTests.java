/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.opensearch.knn.memoryoptsearch.FaissIndexFloatFlatTests.NUM_VECTORS;

public class FaissBinaryHnswIndexTests extends KNNTestCase {
    public static final int CODE_SIZE = 64;
    // 512 binary Dimension
    public static final int BINARY_DIMENSION = (int) (Math.ceil((4 * 128) / 8) * 8);
    private static final int IBHF_START_OFFSET = 25;

    @SneakyThrows
    public void testLoad() {
        // Load binary
        final IndexInput indexInput = loadBinaryHnswIndex();

        // Trigger load
        final FaissIndex faissIndex = FaissIndex.load(indexInput);
        assertTrue(faissIndex instanceof FaissBinaryHnswIndex);
        final FaissBinaryHnswIndex faissBinaryHnswIndex = (FaissBinaryHnswIndex) faissIndex;

        // Validate index
        assertEquals(FaissBinaryHnswIndex.IBHF, faissBinaryHnswIndex.getIndexType());

        // Validate header
        assertEquals(VectorEncoding.BYTE, faissBinaryHnswIndex.getVectorEncoding());
        assertEquals(NUM_VECTORS, faissBinaryHnswIndex.getTotalNumberOfVectors());
        assertEquals(KNNVectorSimilarityFunction.HAMMING, faissBinaryHnswIndex.getVectorSimilarityFunction());
        assertEquals(CODE_SIZE, faissBinaryHnswIndex.getCodeSize());
        assertEquals(BINARY_DIMENSION, faissBinaryHnswIndex.getDimension());
    }

    @SneakyThrows
    private IndexInput loadBinaryHnswIndex() {
        final String relativePath = "data/memoryoptsearch/faiss_binary_50_vectors_512_dim.bin";
        final URL floatFloatVectors = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        byte[] bytes = Files.readAllBytes(Path.of(floatFloatVectors.toURI()));
        bytes = Arrays.copyOfRange(bytes, IBHF_START_OFFSET, bytes.length);
        final IndexInput indexInput = new ByteArrayIndexInput("FaissIndexFloatFlatTests", bytes);
        return indexInput;
    }
}
