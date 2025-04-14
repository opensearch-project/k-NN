/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;
import org.opensearch.knn.memoryoptsearch.faiss.FaissSection;
import org.opensearch.knn.memoryoptsearch.faiss.UnsupportedFaissIndexException;

import java.lang.reflect.Field;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

public class FaissIndexFloatFlatTests extends KNNTestCase {
    static final int NUM_VECTORS = 50;
    static final int DIMENSION = 128;

    @SneakyThrows
    public void testLoad() {
        doTestLoad(FaissIndexFloatFlat.IXF2);
        doTestLoad(FaissIndexFloatFlat.IXFI);
    }

    @SneakyThrows
    public void testLoadInvalidType() {
        final IndexInput indexInput = loadFlatFloatVectors("INVALID_INDEX_TYPE");
        try {
            FaissIndex.load(indexInput);
            fail();
        } catch (UnsupportedFaissIndexException e) {}
    }

    @SneakyThrows
    public void testFloatVectorValues() {
        // Load binary
        IndexInput indexInput = loadFlatFloatVectors(FaissIndexFloatFlat.IXF2);
        final FaissIndex faissIndex = FaissIndexFloatFlat.load(indexInput);

        // Prepare a new input stream
        indexInput = loadFlatFloatVectors(FaissIndexFloatFlat.IXF2);

        // Validate it
        final FloatVectorValues values = faissIndex.getFloatValues(indexInput);
        assertEquals(DIMENSION, values.dimension());
        assertEquals(NUM_VECTORS, values.size());

        float[] vector = values.vectorValue(0);
        assertEquals(DIMENSION, vector.length);
        for (int i = 0; i < DIMENSION; ++i) {
            assertEquals(ANSWER_FIRST_VECTORS[i], vector[i], 1e-3);
        }

        vector = values.vectorValue(NUM_VECTORS - 1);
        for (int i = 0; i < DIMENSION; ++i) {
            assertEquals(ANSWER_LAST_VECTORS[i], vector[i], 1e-3);
        }
    }

    @SneakyThrows
    private static void doTestLoad(final String indexType) {
        // Load binary
        final IndexInput indexInput = loadFlatFloatVectors(indexType);

        // Trigger load
        final FaissIndex faissIndex = FaissIndex.load(indexInput);
        assertTrue(faissIndex instanceof FaissIndexFloatFlat);
        final FaissIndexFloatFlat faissIndexFloatFlat = (FaissIndexFloatFlat) faissIndex;

        // Validate index
        assertEquals(indexType, faissIndexFloatFlat.getIndexType());

        // Code section, we have 50 vectors with 128 dimension
        final Field floatVectorsField = FaissIndexFloatFlat.class.getDeclaredField("floatVectors");
        floatVectorsField.setAccessible(true);
        final FaissSection codesSection = (FaissSection) floatVectorsField.get(faissIndexFloatFlat);
        assertEquals(NUM_VECTORS * DIMENSION * Float.BYTES, codesSection.getSectionSize());

        // One vector size
        final Field oneVectorByteSizeField = FaissIndexFloatFlat.class.getDeclaredField("oneVectorByteSize");
        oneVectorByteSizeField.setAccessible(true);
        final long oneVectorByteSize = (long) oneVectorByteSizeField.get(faissIndexFloatFlat);
        assertEquals(DIMENSION * Float.BYTES, oneVectorByteSize);

        // Encoding
        assertEquals(VectorEncoding.FLOAT32, faissIndexFloatFlat.getVectorEncoding());

        // Similarity function check
        if (indexType.equals(FaissIndexFloatFlat.IXF2)) {
            assertEquals(KNNVectorSimilarityFunction.EUCLIDEAN, faissIndexFloatFlat.getVectorSimilarityFunction());
        } else if (indexType.equals(FaissIndexFloatFlat.IXFI)) {
            assertEquals(KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, faissIndexFloatFlat.getVectorSimilarityFunction());
        }
    }

    @SneakyThrows
    private static IndexInput loadFlatFloatVectors(final String indexType) {
        final String relativePath = "data/memoryoptsearch/faiss_flat_float_50_vectors_128_dim.bin";
        final URL floatFloatVectors = FaissHNSWTests.class.getClassLoader().getResource(relativePath);
        final byte[] bytes = Files.readAllBytes(Path.of(floatFloatVectors.toURI()));
        final byte[] indexTypeFourBytes = indexType.getBytes();
        final byte[] combinedBytes = ByteBuffer.allocate(bytes.length + indexTypeFourBytes.length)
            .put(indexTypeFourBytes)
            .put(bytes)
            .array();
        final IndexInput indexInput = new ByteArrayIndexInput("FaissIndexFloatFlatTests", combinedBytes);
        return indexInput;
    }

    private static final float[] ANSWER_FIRST_VECTORS = new float[] {
        26.6078f,
        61.6245f,
        18.5583f,
        37.3268f,
        24.0666f,
        1.69988f,
        -1.50474f,
        52.1394f,
        26.1416f,
        16.9386f,
        4.50991f,
        42.2415f,
        62.2779f,
        5.70509f,
        46.3046f,
        9.55903f,
        81.6132f,
        74.5831f,
        -3.90388f,
        89.5168f,
        45.8848f,
        -7.67721f,
        86.1319f,
        12.8427f,
        97.9655f,
        3.63787f,
        44.5729f,
        18.8739f,
        41.1438f,
        50.0369f,
        77.694f,
        27.5867f,
        0.114151f,
        41.7555f,
        18.184f,
        78.4843f,
        80.989f,
        -5.35198f,
        28.1574f,
        76.9876f,
        61.8678f,
        69.4719f,
        -7.17243f,
        76.9479f,
        73.8192f,
        18.0001f,
        18.1561f,
        63.9718f,
        61.5573f,
        94.2013f,
        46.9007f,
        73.1804f,
        -5.44495f,
        55.6707f,
        1.61328f,
        11.8573f,
        44.7083f,
        18.4245f,
        90.8937f,
        82.3297f,
        4.8593f,
        65.8558f,
        69.8069f,
        75.2011f,
        94.2877f,
        99.4941f,
        57.8037f,
        80.0829f,
        43.1943f,
        12.252f,
        34.6172f,
        84.0976f,
        19.0696f,
        39.8623f,
        63.8514f,
        9.67021f,
        53.7934f,
        56.4224f,
        -5.10765f,
        37.4656f,
        99.9188f,
        51.9495f,
        16.1503f,
        52.4856f,
        83.4775f,
        75.3885f,
        66.4245f,
        -5.20296f,
        27.9606f,
        23.8542f,
        5.08652f,
        10.9774f,
        -8.84472f,
        0.379398f,
        76.9435f,
        45.0236f,
        91.2635f,
        5.44112f,
        79.1459f,
        62.515f,
        32.7682f,
        94.0194f,
        81.0847f,
        60.6658f,
        69.0975f,
        31.3945f,
        25.6685f,
        77.4493f,
        20.8122f,
        71.7832f,
        6.27018f,
        39.6878f,
        48.6875f,
        25.6163f,
        89.9631f,
        81.3661f,
        -9.12022f,
        34.8925f,
        33.8191f,
        18.5165f,
        43.0279f,
        19.1285f,
        76.3875f,
        82.309f,
        48.5068f,
        52.5117f,
        56.9075f,
        20.0894f };

    private static final float[] ANSWER_LAST_VECTORS = new float[] {
        94.5368f,
        17.7749f,
        92.7226f,
        37.8604f,
        77.2353f,
        5.43306f,
        23.0374f,
        33.7311f,
        81.6744f,
        82.4704f,
        53.4283f,
        90.7504f,
        73.5587f,
        39.8555f,
        49.1712f,
        18.2919f,
        99.4891f,
        55.7042f,
        93.6262f,
        17.8283f,
        55.8974f,
        91.1053f,
        51.7314f,
        52.6911f,
        96.2362f,
        59.9277f,
        6.21873f,
        77.7419f,
        18.0199f,
        83.2658f,
        33.9773f,
        17.7845f,
        90.8755f,
        88.588f,
        55.1007f,
        75.0429f,
        62.7989f,
        42.6708f,
        -3.12951f,
        77.9179f,
        59.8518f,
        23.6182f,
        88.0435f,
        70.8514f,
        61.3155f,
        30.9854f,
        93.902f,
        -9.02647f,
        93.4157f,
        59.6133f,
        -8.96786f,
        61.7919f,
        63.8478f,
        23.9899f,
        61.5031f,
        5.40641f,
        75.0171f,
        59.1598f,
        35.6085f,
        7.92753f,
        5.54097f,
        3.32967f,
        64.9688f,
        37.1329f,
        98.8765f,
        -7.02708f,
        55.7849f,
        61.3619f,
        66.4946f,
        -1.76433f,
        35.9853f,
        65.5138f,
        8.17353f,
        2.32414f,
        88.2795f,
        12.6217f,
        89.7036f,
        69.9144f,
        96.955f,
        94.2342f,
        61.7649f,
        27.1929f,
        58.0297f,
        67.8765f,
        18.7195f,
        97.4808f,
        52.913f,
        17.2338f,
        15.1667f,
        68.0844f,
        44.9571f,
        62.1126f,
        19.7982f,
        38.0412f,
        87.9569f,
        49.1288f,
        13.712f,
        29.5886f,
        66.9289f,
        68.8291f,
        22.5455f,
        -9.96551f,
        -3.53541f,
        95.3707f,
        70.3642f,
        34.3594f,
        41.085f,
        45.6985f,
        2.88102f,
        47.8573f,
        78.9176f,
        53.2515f,
        40.1408f,
        43.7638f,
        24.3534f,
        72.0761f,
        16.7687f,
        -6.16205f,
        36.2928f,
        60.762f,
        82.9339f,
        83.7456f,
        42.1101f,
        30.3755f,
        -7.78363f,
        67.233f,
        80.6584f,
        27.1251f };
}
