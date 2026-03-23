/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.knn.KNNTestCase;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class VectorScorerModeTests extends KNNTestCase {

    // ──────────────────────────────────────────────
    // SCORE mode
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testScore_withFloatVectorValues_delegatesToScorer() {
        float[] target = { 1.0f, 2.0f };
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        VectorScorer expected = mock(VectorScorer.class);
        when(floatVectorValues.scorer(target)).thenReturn(expected);

        VectorScorer result = VectorScorerMode.SCORE.createScorer(floatVectorValues, target);

        assertSame(expected, result);
        verify(floatVectorValues).scorer(target);
    }

    @SneakyThrows
    public void testScore_withByteVectorValues_delegatesToScorer() {
        byte[] target = { 1, 2 };
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        VectorScorer expected = mock(VectorScorer.class);
        when(byteVectorValues.scorer(target)).thenReturn(expected);

        VectorScorer result = VectorScorerMode.SCORE.createScorer(byteVectorValues, target);

        assertSame(expected, result);
        verify(byteVectorValues).scorer(target);
    }

    // ──────────────────────────────────────────────
    // RESCORE mode
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testRescore_withFloatVectorValues_delegatesToRescorer() {
        float[] target = { 1.0f, 2.0f };
        FloatVectorValues floatVectorValues = mock(FloatVectorValues.class);
        VectorScorer expected = mock(VectorScorer.class);
        when(floatVectorValues.rescorer(target)).thenReturn(expected);

        VectorScorer result = VectorScorerMode.RESCORE.createScorer(floatVectorValues, target);

        assertSame(expected, result);
        verify(floatVectorValues).rescorer(target);
    }

    @SneakyThrows
    public void testRescore_withByteVectorValues_delegatesToRescorer() {
        byte[] target = { 1, 2 };
        ByteVectorValues byteVectorValues = mock(ByteVectorValues.class);
        VectorScorer expected = mock(VectorScorer.class);
        when(byteVectorValues.rescorer(target)).thenReturn(expected);

        VectorScorer result = VectorScorerMode.RESCORE.createScorer(byteVectorValues, target);

        assertSame(expected, result);
        verify(byteVectorValues).rescorer(target);
    }
}
