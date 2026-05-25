/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import junit.framework.TestCase;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.mockito.MockedConstruction;
import org.mockito.MockedStatic;
import org.opensearch.knn.memoryoptsearch.faiss.MMapFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.MMapVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.NativeRandomVectorScorer;
import org.opensearch.knn.memoryoptsearch.faiss.WrappedFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

public class NativeEngines990KnnVectorsScorerTests extends TestCase {

    private final FlatVectorsScorer delegate = mock(FlatVectorsScorer.class);
    private final NativeEngines990KnnVectorsScorer scorer = new NativeEngines990KnnVectorsScorer(delegate);

    public void testFloatVector_mmapValues_euclidean_returnsNativeScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        FloatVectorValues mmapValues = mock(FloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }

    public void testFloatVector_mmapValues_maxInnerProduct_returnsNativeScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        FloatVectorValues mmapValues = mock(FloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }

    public void testFloatVector_nonMmapValues_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        FloatVectorValues plainValues = mock(FloatVectorValues.class);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        try (MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(plainValues);
            when(delegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target)).thenReturn(expectedScorer);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertSame(expectedScorer, result);
        }
    }

    public void testFloatVector_unsupportedSimilarity_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        when(delegate.getRandomVectorScorer(VectorSimilarityFunction.COSINE, vectorValues, target)).thenReturn(expectedScorer);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.COSINE, vectorValues, target);

        assertSame(expectedScorer, result);
    }

    public void testFloatVector_dotProduct_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        when(delegate.getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, vectorValues, target)).thenReturn(expectedScorer);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, vectorValues, target);

        assertSame(expectedScorer, result);
    }

    public void testFloatVector_nullBottomValues_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        try (MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(null);
            when(delegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target)).thenReturn(expectedScorer);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertSame(expectedScorer, result);
        }
    }

    public void testByteVector_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        byte[] target = new byte[] { 1, 2 };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        when(delegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target)).thenReturn(expectedScorer);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

        assertSame(expectedScorer, result);
    }

    public void testScorerSupplier_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        RandomVectorScorerSupplier expectedSupplier = mock(RandomVectorScorerSupplier.class);

        when(delegate.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues)).thenReturn(expectedSupplier);

        RandomVectorScorerSupplier result = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues);

        assertSame(expectedSupplier, result);
        verify(delegate).getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues);
    }

    // BF16 tests

    public void testFloatVector_bf16MmapValues_euclidean_returnsNativeScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(FaissQuantizerType.QT_BF16);
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }

    public void testFloatVector_bf16MmapValues_maxInnerProduct_returnsNativeScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(FaissQuantizerType.QT_BF16);
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }

    public void testFloatVector_bf16MmapValues_cosine_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(FaissQuantizerType.QT_BF16);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        try (MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);
            when(delegate.getRandomVectorScorer(VectorSimilarityFunction.COSINE, vectorValues, target)).thenReturn(expectedScorer);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.COSINE, vectorValues, target);

            assertSame(expectedScorer, result);
        }
    }

    public void testFloatVector_bf16MmapValues_dotProduct_delegatesToWrapped() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(FaissQuantizerType.QT_BF16);
        float[] target = new float[] { 1.0f, 2.0f };
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);

        try (MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);
            when(delegate.getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, vectorValues, target)).thenReturn(expectedScorer);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.DOT_PRODUCT, vectorValues, target);

            assertSame(expectedScorer, result);
        }
    }

    public void testFloatVector_fp16MmapFloatValues_euclidean_returnsNativeScorer() throws IOException {
        // MMapFloatVectorValues with FP16 quantizer type (not BF16) should still use the FP16 native path
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(FaissQuantizerType.QT_FP16);
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }

    public void testFloatVector_mmapFloatValues_nullReconstructor_euclidean_returnsNativeScorer() throws IOException {
        // MMapFloatVectorValues with null quantizer type should use the FP16 path (isBF16 = false)
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        MMapFloatVectorValues mmapFloatValues = mock(MMapFloatVectorValues.class, withSettings().extraInterfaces(MMapVectorValues.class));
        when(((MMapVectorValues) mmapFloatValues).getAddressAndSize()).thenReturn(new long[] { 100L, 200L });
        when(mmapFloatValues.getQuantizerType()).thenReturn(null);
        float[] target = new float[] { 1.0f, 2.0f };

        try (
            MockedConstruction<NativeRandomVectorScorer> nativeScorerMock = mockConstruction(NativeRandomVectorScorer.class);
            MockedStatic<WrappedFloatVectorValues> wrappedMock = mockStatic(WrappedFloatVectorValues.class)
        ) {
            wrappedMock.when(() -> WrappedFloatVectorValues.getBottomFloatVectorValues(vectorValues)).thenReturn(mmapFloatValues);

            RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

            assertNotNull(result);
            assertTrue(result instanceof NativeRandomVectorScorer);
            verifyNoInteractions(delegate);
        }
    }
}
