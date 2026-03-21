/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.faiss.SQConfig;
import org.opensearch.knn.index.engine.faiss.SQConfigParser;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class FlatVectorsScorerResolverTests extends KNNTestCase {

    private static final FlatVectorsScorer DELEGATE_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    // ---- AdcScorerResolver ----

    public void testAdcCanResolve_returnsTrueForAdcField() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.AdcScorerResolver();
        final FieldInfo fieldInfo = mockAdcField(SpaceType.L2);
        assertTrue(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN));
    }

    public void testAdcCanResolve_returnsFalseForNonAdcField() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.AdcScorerResolver();
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        assertFalse(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN));
    }

    public void testAdcResolve_returnsCorrectScorerForEachSpaceType() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.AdcScorerResolver();

        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL }) {
            final FieldInfo fieldInfo = mockAdcField(spaceType);
            final FlatVectorsScorer scorer = resolver.resolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, DELEGATE_SCORER);
            assertNotNull("Expected non-null scorer for space type: " + spaceType, scorer);
            assertNotSame("Expected ADC scorer, not delegate, for space type: " + spaceType, DELEGATE_SCORER, scorer);
            assertSame(
                "Expected same cached instance for repeated resolve of space type: " + spaceType,
                scorer,
                resolver.resolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, DELEGATE_SCORER)
            );
        }
    }

    // ---- HammingScorerResolver ----

    public void testHammingCanResolve_returnsTrueForHammingSimilarityFunction() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.HammingScorerResolver();
        assertTrue(resolver.canResolve(mock(FieldInfo.class), KNNVectorSimilarityFunction.HAMMING));
    }

    public void testHammingCanResolve_returnsFalseForNonHammingSimilarityFunction() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.HammingScorerResolver();
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        assertFalse(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN));
        assertFalse(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.COSINE));
        assertFalse(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT));
    }

    public void testHammingResolve_returnsSameCachedInstance() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.HammingScorerResolver();
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final FlatVectorsScorer first = resolver.resolve(fieldInfo, KNNVectorSimilarityFunction.HAMMING, DELEGATE_SCORER);
        final FlatVectorsScorer second = resolver.resolve(fieldInfo, KNNVectorSimilarityFunction.HAMMING, DELEGATE_SCORER);
        assertNotNull(first);
        assertSame("Expected same cached HammingFlatVectorsScorer instance", first, second);
        assertNotSame("Expected Hamming scorer, not delegate", DELEGATE_SCORER, first);
    }

    // ---- FaissSQScorerResolver ----

    public void testFaissSQCanResolve_returnsTrueForOneBitSQField() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.FaissSQScorerResolver();
        final FieldInfo fieldInfo = mockSQField(FaissSQEncoder.Bits.ONE.getValue());
        assertTrue(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN));
    }

    public void testFaissSQCanResolve_returnsFalseForNonOneBitSQField() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.FaissSQScorerResolver();
        final FieldInfo fieldInfo = mockSQField(2);
        assertFalse(resolver.canResolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN));
    }

    public void testFaissSQCanResolve_returnsFalseForNonSQField() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.FaissSQScorerResolver();
        assertFalse(resolver.canResolve(mock(FieldInfo.class), KNNVectorSimilarityFunction.EUCLIDEAN));
    }

    public void testFaissSQResolve_wrapsDelegate() {
        final FlatVectorsScorerResolver resolver = new FlatVectorsScorerResolver.FaissSQScorerResolver();
        final FieldInfo fieldInfo = mockSQField(FaissSQEncoder.Bits.ONE.getValue());
        final FlatVectorsScorer scorer = resolver.resolve(fieldInfo, KNNVectorSimilarityFunction.EUCLIDEAN, DELEGATE_SCORER);
        assertNotNull(scorer);
        assertNotSame("Expected a wrapping SQ scorer, not the delegate itself", DELEGATE_SCORER, scorer);
    }

    // ---- Helpers ----

    private static FieldInfo mockAdcField(SpaceType spaceType) {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String adcConfig = QuantizationConfigParser.toCsv(
            QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).enableADC(true).build()
        );
        when(fieldInfo.getAttribute(KNNConstants.QFRAMEWORK_CONFIG)).thenReturn(adcConfig);
        when(fieldInfo.getAttribute(KNNConstants.SPACE_TYPE)).thenReturn(spaceType.getValue());
        return fieldInfo;
    }

    private static FieldInfo mockSQField(int bits) {
        final FieldInfo fieldInfo = mock(FieldInfo.class);
        final String sqConfig = SQConfigParser.toCsv(SQConfig.builder().bits(bits).build());
        when(fieldInfo.getAttribute(KNNConstants.SQ_CONFIG)).thenReturn(sqConfig);
        return fieldInfo;
    }
}
