/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyByte;
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

        // Set up quantization mocks needed by the parent Lucene104ScalarQuantizedVectorScorer
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(ScalarEncoding.UNSIGNED_BYTE);

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
     * Exercises the {@code IllegalStateException} guard inside {@code bulkSimdRandomVectorScorer}:
     * when the address extractor returns a non-null address (so the SIMD path is entered) but the
     * scalar encoding is not {@code SINGLE_BIT_QUERY_NIBBLE}, the method must throw.
     */
    @SneakyThrows
    public void testGetRandomVectorScorer_whenScalarEncodingIsNotSingleBitQueryNibble_thenThrowsIllegalStateException() {
        final FlatVectorsScorer mockDelegate = mock(FlatVectorsScorer.class);
        final KNN1040ScalarQuantizedVectorScorer scorer = new KNN1040ScalarQuantizedVectorScorer(mockDelegate);

        final int dimension = 8;
        final QuantizedByteVectorValues mockQuantizedValues = mock(QuantizedByteVectorValues.class);
        final IndexInput mockIndexInput = mock(IndexInput.class);
        when(mockQuantizedValues.getSlice()).thenReturn(mockIndexInput);
        when(mockIndexInput.length()).thenReturn(1024L);
        when(mockQuantizedValues.dimension()).thenReturn(dimension);
        // Use an encoding that is NOT SINGLE_BIT_QUERY_NIBBLE to trigger the guard.
        when(mockQuantizedValues.getScalarEncoding()).thenReturn(ScalarEncoding.UNSIGNED_BYTE);

        final StubVectorValues stub = new StubVectorValues();
        final java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, mockQuantizedValues);

        final float[] target = new float[dimension];
        final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        // Mock the extractor to return a non-null addressAndSize so the SIMD path is entered.
        try (MockedStatic<MemorySegmentAddressExtractorUtil> mockedStatic = Mockito.mockStatic(MemorySegmentAddressExtractorUtil.class)) {
            mockedStatic.when(
                () -> MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    any(IndexInput.class),
                    Mockito.anyLong(),
                    Mockito.anyLong()
                )
            ).thenReturn(new long[] { 0L, 1024L });

            final IllegalStateException ex = expectThrows(
                IllegalStateException.class,
                () -> scorer.getRandomVectorScorer(similarityFunction, stub, target)
            );
            assertTrue(
                "Exception message should mention SINGLE_BIT_QUERY_NIBBLE, was: " + ex.getMessage(),
                ex.getMessage().contains("SINGLE_BIT_QUERY_NIBBLE")
            );
        }
    }
}
