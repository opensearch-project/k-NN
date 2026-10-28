/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.mockito.ArgumentCaptor;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyByte;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Log4j2
public class KNN1040ScalarQuantizedVectorScorerTests extends KNNTestCase {

    /**
     * A concrete stub extending KnnVectorValues that declares the private field
     * {@code quantizedVectorValues} so that reflection-based extraction succeeds.
     */
    static class StubVectorValues extends KnnVectorValues {
        private QuantizedByteVectorValues quantizedVectorValues;

        @Override
        public int dimension() {
            return 0;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public KnnVectorValues copy() throws IOException {
            return this;
        }

        @Override
        public VectorEncoding getEncoding() {
            return VectorEncoding.FLOAT32;
        }
    }

    /**
     * For 2-bit encodings the scorer must route to Lucene's reference scorer without touching
     * {@code MemorySegmentAddressExtractorUtil} — native bulk SIMD is not wired for 2-bit yet.
     */
    @SneakyThrows
    private void assertMultiBitFallback(final ScalarEncoding encoding) {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040ScalarQuantizedVectorScorer scorer = new KNN1040ScalarQuantizedVectorScorer(mockDelegate);

        final int dimension = 8;
        final QuantizedByteVectorValues mockQuantizedValues = mock(QuantizedByteVectorValues.class);
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(encoding);

        // Parent Lucene104ScalarQuantizedVectorScorer's reference path also runs a quantization —
        // supply the same mocks the null-address test uses so it can complete without NPEs.
        final OptimizedScalarQuantizer mockQuantizer = mock(OptimizedScalarQuantizer.class);
        when(mockQuantizedValues.getQuantizer()).thenReturn(mockQuantizer);
        when(mockQuantizedValues.getCentroid()).thenReturn(new float[dimension]);
        final OptimizedScalarQuantizer.QuantizationResult quantizationResult = new OptimizedScalarQuantizer.QuantizationResult(
            0.0f,
            1.0f,
            0.0f,
            0
        );
        when(mockQuantizer.scalarQuantize(any(float[].class), any(byte[].class), anyByte(), any(float[].class))).thenReturn(
            quantizationResult
        );

        final StubVectorValues stub = new StubVectorValues();
        final java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, mockQuantizedValues);

        try (MockedStatic<MemorySegmentAddressExtractorUtil> mockedStatic = Mockito.mockStatic(MemorySegmentAddressExtractorUtil.class)) {
            final RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, stub, new float[dimension]);

            assertNotNull(result);
            // The multi-bit gate must fire before we even ask about the address — no interaction.
            mockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_whenTwoBitEncoding_thenRoutesToLuceneScorerWithoutAddressExtraction() {
        assertMultiBitFallback(ScalarEncoding.DIBIT_QUERY_NIBBLE);
    }

    /**
     * For encodings with a native kernel (1-bit and 4-bit) and a resolvable memory address, the scorer must push
     * the query into the native search context with the matching similarity function type. 1-bit queries are
     * transposed into 4 bit planes (packed, half the discretized dimension in bytes) while 4-bit queries stay
     * unpacked (one nibble per byte).
     */
    @SneakyThrows
    private void assertNativeRouting(
        final ScalarEncoding encoding,
        final VectorSimilarityFunction similarityFunction,
        final SimdVectorComputeService.SimilarityFunctionType expectedFunctionType,
        final int expectedQueryLength
    ) {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040ScalarQuantizedVectorScorer scorer = new KNN1040ScalarQuantizedVectorScorer(mockDelegate);

        final int dimension = 8;
        final QuantizedByteVectorValues mockQuantizedValues = mock(QuantizedByteVectorValues.class);
        final IndexInput mockIndexInput = mock(IndexInput.class);
        when(mockQuantizedValues.getSlice()).thenReturn(mockIndexInput);
        when(mockIndexInput.length()).thenReturn(1024L);
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(encoding);
        when(mockQuantizedValues.getCentroid()).thenReturn(new float[dimension]);
        when(mockQuantizedValues.getCentroidDP()).thenReturn(0.5f);

        final OptimizedScalarQuantizer mockQuantizer = mock(OptimizedScalarQuantizer.class);
        when(mockQuantizedValues.getQuantizer()).thenReturn(mockQuantizer);
        when(mockQuantizer.scalarQuantize(any(float[].class), any(byte[].class), anyByte(), any(float[].class))).thenReturn(
            new OptimizedScalarQuantizer.QuantizationResult(0.0f, 1.0f, 0.0f, 0)
        );

        final StubVectorValues stub = new StubVectorValues();
        final java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, mockQuantizedValues);

        // Unit-length query so the cosine normalization assertion holds.
        final float[] target = new float[dimension];
        target[0] = 1.0f;

        try (
            MockedStatic<MemorySegmentAddressExtractorUtil> mockedExtractor = Mockito.mockStatic(MemorySegmentAddressExtractorUtil.class);
            MockedStatic<SimdVectorComputeService> mockedSimd = Mockito.mockStatic(SimdVectorComputeService.class)
        ) {
            mockedExtractor.when(
                () -> MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(any(IndexInput.class), anyLong(), anyLong())
            ).thenReturn(new long[] { 0L, 1024L });

            final RandomVectorScorer result = scorer.getRandomVectorScorer(similarityFunction, stub, target);
            assertNotNull(result);

            final ArgumentCaptor<byte[]> queryCaptor = ArgumentCaptor.forClass(byte[].class);
            mockedSimd.verify(
                () -> SimdVectorComputeService.saveSQSearchContext(
                    queryCaptor.capture(),
                    eq(0.0f),
                    eq(1.0f),
                    eq(0.0f),
                    eq(0),
                    any(long[].class),
                    eq(expectedFunctionType.ordinal()),
                    eq(dimension),
                    eq(0.5f)
                )
            );
            assertEquals(expectedQueryLength, queryCaptor.getValue().length);
        }
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_whenFourBitEncoding_thenRoutesToNativeScorer() {
        // 4-bit query is unpacked: one byte per (discretized) dimension.
        assertNativeRouting(
            ScalarEncoding.PACKED_NIBBLE,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SimdVectorComputeService.SimilarityFunctionType.SQ4_IP,
            8
        );
        assertNativeRouting(
            ScalarEncoding.PACKED_NIBBLE,
            VectorSimilarityFunction.EUCLIDEAN,
            SimdVectorComputeService.SimilarityFunctionType.SQ4_L2,
            8
        );
        assertNativeRouting(
            ScalarEncoding.PACKED_NIBBLE,
            VectorSimilarityFunction.COSINE,
            SimdVectorComputeService.SimilarityFunctionType.SQ4_COSINE,
            8
        );
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_whenOneBitEncoding_thenRoutesToNativeScorer() {
        // 1-bit query is transposed into 4 bit planes: 4 bits * 8 dims / 8 = 4 bytes.
        assertNativeRouting(
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            SimdVectorComputeService.SimilarityFunctionType.SQ_IP,
            4
        );
        assertNativeRouting(
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            VectorSimilarityFunction.EUCLIDEAN,
            SimdVectorComputeService.SimilarityFunctionType.SQ_L2,
            4
        );
        assertNativeRouting(
            ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE,
            VectorSimilarityFunction.COSINE,
            SimdVectorComputeService.SimilarityFunctionType.SQ_COSINE,
            4
        );
    }

    @SneakyThrows
    public void testGetRandomVectorScorer_whenAddressExtractionReturnsNull_thenFallsBackToLuceneScorer() {
        // Arrange: set up the mock delegate scorer
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);

        // Create the scorer under test
        final KNN1040ScalarQuantizedVectorScorer scorer = new KNN1040ScalarQuantizedVectorScorer(mockDelegate);

        // Set up QuantizedByteVectorValues mock with all methods needed by the parent class
        final int dimension = 8;
        final QuantizedByteVectorValues mockQuantizedValues = mock(QuantizedByteVectorValues.class);
        final IndexInput mockIndexInput = mock(IndexInput.class);
        when(mockQuantizedValues.getSlice()).thenReturn(mockIndexInput);
        when(mockIndexInput.length()).thenReturn(1024L);

        // Set up quantization mocks needed by the parent Lucene104ScalarQuantizedVectorScorer.
        // Encoding is SINGLE_BIT_QUERY_NIBBLE (docBits == 1) so the multi-bit fallback gate does
        // not fire — this test exercises the address-extractor null fallback specifically.
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);

        final OptimizedScalarQuantizer mockQuantizer = mock(OptimizedScalarQuantizer.class);
        when(mockQuantizedValues.getQuantizer()).thenReturn(mockQuantizer);
        when(mockQuantizedValues.getCentroid()).thenReturn(new float[dimension]);

        // The parent's quantization path calls scalarQuantize — return a valid result
        final OptimizedScalarQuantizer.QuantizationResult quantizationResult = new OptimizedScalarQuantizer.QuantizationResult(
            0.0f,
            1.0f,
            0.0f,
            0
        );
        when(mockQuantizer.scalarQuantize(any(float[].class), any(byte[].class), anyByte(), any(float[].class))).thenReturn(
            quantizationResult
        );

        // Create stub with the quantizedVectorValues field set via reflection
        final StubVectorValues stub = new StubVectorValues();
        final java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, mockQuantizedValues);

        final float[] target = new float[dimension];
        final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        // Mock MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize to return null (triggers fallback)
        try (MockedStatic<MemorySegmentAddressExtractorUtil> mockedStatic = Mockito.mockStatic(MemorySegmentAddressExtractorUtil.class)) {
            mockedStatic.when(
                () -> MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    any(IndexInput.class),
                    Mockito.anyLong(),
                    Mockito.anyLong()
                )
            ).thenReturn(null);

            // Act
            final RandomVectorScorer result = scorer.getRandomVectorScorer(similarityFunction, stub, target);

            // Assert: the fallback path returns a non-null scorer without throwing
            assertNotNull(result);
        }
    }

    /**
     * Exercises the encoding guard in {@code getScorer}: the scalar encoding is resolved via
     * {@link ScalarEncodingResolver#docBits} to decide the scoring path (1/4-bit → native SIMD,
     * 2-bit → Lucene fallback). An encoding whose bit width is outside the supported set
     * (here {@code UNSIGNED_BYTE}, 8 bits) is rejected up front with an
     * {@link IllegalArgumentException} before any scorer is built.
     */
    @SneakyThrows
    public void testGetRandomVectorScorer_whenScalarEncodingBitWidthUnsupported_thenThrowsIllegalArgumentException() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040ScalarQuantizedVectorScorer scorer = new KNN1040ScalarQuantizedVectorScorer(mockDelegate);

        final int dimension = 8;
        final QuantizedByteVectorValues mockQuantizedValues = mock(QuantizedByteVectorValues.class);
        final IndexInput mockIndexInput = mock(IndexInput.class);
        when(mockQuantizedValues.getSlice()).thenReturn(mockIndexInput);
        when(mockIndexInput.length()).thenReturn(1024L);
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        // Use an encoding whose bit width (8) is not one of the supported FAISS SQ widths (1, 2, 4)
        // to trigger the ScalarEncodingResolver#docBits guard.
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(ScalarEncoding.UNSIGNED_BYTE);

        final StubVectorValues stub = new StubVectorValues();
        final java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, mockQuantizedValues);

        final float[] target = new float[dimension];
        final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        // Mock the extractor to return a non-null addressAndSize so, absent the encoding guard,
        // the SIMD path would be entered.
        try (MockedStatic<MemorySegmentAddressExtractorUtil> mockedStatic = Mockito.mockStatic(MemorySegmentAddressExtractorUtil.class)) {
            mockedStatic.when(
                () -> MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    any(IndexInput.class),
                    Mockito.anyLong(),
                    Mockito.anyLong()
                )
            ).thenReturn(new long[] { 0L, 1024L });

            final IllegalArgumentException ex = expectThrows(
                IllegalArgumentException.class,
                () -> scorer.getRandomVectorScorer(similarityFunction, stub, target)
            );
            assertTrue(
                "Exception message should mention the unsupported encoding, was: " + ex.getMessage(),
                ex.getMessage().contains("Unsupported SQ scalar encoding") && ex.getMessage().contains("UNSIGNED_BYTE")
            );
        }
    }
}
