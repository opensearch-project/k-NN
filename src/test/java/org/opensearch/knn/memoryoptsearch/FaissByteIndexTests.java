/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexScalarQuantizedFlat;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructor;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizedValueReconstructorFactory;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.mockito.Mockito.mock;

public class FaissByteIndexTests extends KNNTestCase {
    static final int NUM_VECTORS = 100;
    static final int DIMENSION = 8;

    @SneakyThrows
    public void testLoad() {
        // Load binary
        final IndexInput indexInput = loadFlatByteVectors();
        final FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);

        // Trigger load
        final FaissIndex faissIndex = FaissIdMapIndex.load(indexInput, flatVectorsReaderWithFieldName);
        assertTrue(faissIndex instanceof FaissIndexScalarQuantizedFlat);
        final FaissIndexScalarQuantizedFlat faissByteIndex = (FaissIndexScalarQuantizedFlat) faissIndex;

        // Encoding type
        assertEquals(VectorEncoding.BYTE, faissByteIndex.getVectorEncoding());

        // Validate index
        assertEquals(FaissIndexScalarQuantizedFlat.IXSQ, faissByteIndex.getIndexType());

        // Quantizer type
        assertEquals(FaissQuantizerType.QT_8BIT_DIRECT_SIGNED, faissByteIndex.getQuantizerType());

        // Range check
        assertEquals(FaissIndexScalarQuantizedFlat.RangeStat.MIN_MAX, faissByteIndex.getRangeStat());

        // Dimensions
        assertEquals(DIMENSION, faissByteIndex.getDimension());

        // Vector size
        assertEquals(DIMENSION, faissByteIndex.getOneVectorByteSize());
        // Each element is one single byte, hence 8 bits.
        assertEquals(8, faissByteIndex.getOneVectorElementBits());

        // 0th vector validation
        final byte[] decodedBytes = new byte[DIMENSION];
        final FaissQuantizedValueReconstructor decoder = FaissQuantizedValueReconstructorFactory.create(
            FaissQuantizerType.QT_8BIT_DIRECT_SIGNED,
            DIMENSION,
            8 * Byte.BYTES
        );
        decoder.reconstruct(ANSWER_FIRST_QUANTIZED_VECTORS, decodedBytes);
        ByteVectorValues byteVectorValues = faissByteIndex.getByteValues(indexInput);
        assertArrayEquals(decodedBytes, byteVectorValues.vectorValue(0));

        // Last vector validation
        byteVectorValues = faissByteIndex.getByteValues(indexInput);
        decoder.reconstruct(ANSWER_LAST_QUANTIZED_VECTORS, decodedBytes);
        assertArrayEquals(decodedBytes, byteVectorValues.vectorValue(NUM_VECTORS - 1));
    }

    @SneakyThrows
    private static IndexInput loadFlatByteVectors() {
        final String relativePath = "data/memoryoptsearch/faiss_flat_byte_100_vectors_8_dim.bin";
        final URL flatByteVectors = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        final byte[] indexTypeFourBytes = FaissIndexScalarQuantizedFlat.IXSQ.getBytes();
        final byte[] bytes = Files.readAllBytes(Path.of(flatByteVectors.toURI()));
        final byte[] combinedBytes = ByteBuffer.allocate(bytes.length + indexTypeFourBytes.length)
            .put(indexTypeFourBytes)
            .put(bytes)
            .array();
        final IndexInput indexInput = new ByteArrayIndexInput("FaissByteIndexTests", combinedBytes);
        return indexInput;
    }

    private static final byte[] ANSWER_FIRST_QUANTIZED_VECTORS = new byte[] {
        (byte) 183,
        (byte) 162,
        (byte) 169,
        (byte) 169,
        (byte) 178,
        (byte) 214,
        (byte) 220,
        (byte) 201, };

    private static final byte[] ANSWER_LAST_QUANTIZED_VECTORS = new byte[] {
        (byte) 182,
        (byte) 148,
        (byte) 195,
        (byte) 139,
        (byte) 150,
        (byte) 184,
        (byte) 189,
        (byte) 221 };
}
