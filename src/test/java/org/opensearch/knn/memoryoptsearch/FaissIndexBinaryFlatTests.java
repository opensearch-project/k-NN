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
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;
import org.opensearch.knn.memoryoptsearch.faiss.FaissSection;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.lang.reflect.Field;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.mockito.Mockito.mock;

public class FaissIndexBinaryFlatTests extends KNNTestCase {
    public static final int IBXF_START_OFFSET = 7474;
    public static final int NUM_VECTORS = 50;
    public static final int CODE_SIZE = 64;
    // 512 binary Dimension
    public static final int BINARY_DIMENSION = (int) (Math.ceil((4 * 128) / 8) * 8);

    @SneakyThrows
    public void testLoad() {
        // Load binary
        final IndexInput indexInput = loadFlatBinaryVectors();
        final FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);

        // Trigger load
        final FaissIndex faissIndex = FaissIndex.load(indexInput, flatVectorsReaderWithFieldName);
        assertTrue(faissIndex instanceof FaissIndexBinaryFlat);
        final FaissIndexBinaryFlat faissIndexBinaryFlat = (FaissIndexBinaryFlat) faissIndex;

        // Validate index
        assertEquals(FaissIndexBinaryFlat.IBXF, faissIndexBinaryFlat.getIndexType());

        // Code section, we have 50 vectors with 512 dimension with code size 64
        // e.g. 8x compression is applied to 128 float dimension.
        // 512 dimension = Ceil((4 * 128) / 8) * 8
        final Field binaryFlatVectorSectionField = FaissIndexBinaryFlat.class.getDeclaredField("binaryFlatVectorSection");
        binaryFlatVectorSectionField.setAccessible(true);
        final FaissSection codesSection = (FaissSection) binaryFlatVectorSectionField.get(faissIndexBinaryFlat);
        assertEquals(NUM_VECTORS * CODE_SIZE, codesSection.getSectionSize());

        // Validate header
        assertEquals(VectorEncoding.BYTE, faissIndexBinaryFlat.getVectorEncoding());
        assertEquals(NUM_VECTORS, faissIndexBinaryFlat.getTotalNumberOfVectors());
        assertEquals(KNNVectorSimilarityFunction.HAMMING, faissIndexBinaryFlat.getVectorSimilarityFunction());
        assertEquals(CODE_SIZE, faissIndexBinaryFlat.getCodeSize());
        assertEquals(BINARY_DIMENSION, faissIndexBinaryFlat.getDimension());
    }

    @SneakyThrows
    public void testByteVectorValues() {
        // Load binary
        IndexInput indexInput = loadFlatBinaryVectors();
        final FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);
        final FaissIndex faissIndex = FaissIndexFloatFlat.load(indexInput, flatVectorsReaderWithFieldName);

        // Prepare a new input stream
        indexInput = loadFlatBinaryVectors();

        // Validate dimension and size
        final ByteVectorValues values = faissIndex.getByteValues(indexInput);
        assertEquals(BINARY_DIMENSION, values.dimension());
        assertEquals(NUM_VECTORS, values.size());

        // Validate the first vector values
        byte[] vector = values.vectorValue(0);
        assertArrayEquals(FIRST_VECTOR, vector);

        // Validate the last vector values
        vector = values.vectorValue(values.size() - 1);
        assertArrayEquals(LAST_VECTOR, vector);
    }

    @SneakyThrows
    private IndexInput loadFlatBinaryVectors() {
        final String relativePath = "data/memoryoptsearch/faiss_binary_50_vectors_512_dim.bin";
        final URL binaryIndexUrl = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        byte[] bytes = Files.readAllBytes(Path.of(binaryIndexUrl.toURI()));
        bytes = Arrays.copyOfRange(bytes, IBXF_START_OFFSET, bytes.length);
        final IndexInput indexInput = new ByteArrayIndexInput("FaissIndexFloatFlatTests", bytes);
        return indexInput;
    }

    @SneakyThrows
    public void testFloatVectorValuesNotSupported() {
        // Load binary
        final IndexInput indexInput = loadFlatBinaryVectors();
        final FlatVectorsReaderWithFieldName flatVectorsReaderWithFieldName = mock(FlatVectorsReaderWithFieldName.class);
        // Trigger load
        final FaissIndex faissIndex = FaissIndex.load(indexInput, flatVectorsReaderWithFieldName);

        // Try to get float values
        try {
            faissIndex.getFloatValues(indexInput);
            fail();
        } catch (UnsupportedOperationException e) {
            // Expected
        }
    }

    private static final byte[] FIRST_VECTOR = new byte[] {
        75,
        -23,
        124,
        60,
        123,
        19,
        -50,
        -37,
        81,
        -116,
        -67,
        -1,
        73,
        107,
        -75,
        -41,
        74,
        -87,
        40,
        48,
        122,
        19,
        -50,
        -37,
        81,
        -116,
        -67,
        -1,
        64,
        107,
        52,
        -41,
        74,
        -87,
        40,
        16,
        122,
        17,
        70,
        -38,
        16,
        -116,
        -71,
        -33,
        64,
        65,
        20,
        -42,
        72,
        -87,
        40,
        0,
        66,
        17,
        70,
        -118,
        16,
        0,
        -80,
        87,
        64,
        0,
        0,
        -60 };
    private static final byte[] LAST_VECTOR = new byte[] {
        -13,
        -2,
        78,
        106,
        -19,
        -10,
        123,
        -13,
        127,
        -35,
        127,
        -115,
        -10,
        -13,
        57,
        -5,
        115,
        62,
        70,
        104,
        -116,
        -58,
        91,
        -45,
        111,
        -55,
        95,
        -115,
        -74,
        -15,
        57,
        -14,
        114,
        14,
        70,
        104,
        -124,
        -58,
        26,
        -47,
        79,
        -55,
        87,
        -116,
        -90,
        113,
        49,
        -110,
        50,
        10,
        6,
        8,
        -124,
        70,
        18,
        -48,
        79,
        -120,
        6,
        -116,
        -96,
        112,
        33,
        18 };
}
