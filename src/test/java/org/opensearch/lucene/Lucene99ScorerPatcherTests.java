/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.codec.scorer.PrefetchableScorerTestUtils;

/**
 * Tests for {@link Lucene99ScorerPatcher}.
 *
 * <p>{@code installOnce()} mutates process-wide state: the {@code FlatVectorsScorer} on the single shared
 * {@code Lucene99FlatVectorsFormat} held by {@link Lucene99HnswVectorsFormat}, plus a static {@code installed}
 * flag on the patcher. Each test therefore resets both around itself (via {@link PrefetchableScorerTestUtils}) so it
 * does not depend on (or pollute) other tests. When the JVM forbids mutating the {@code private final} scorer field
 * the patch cannot work at all, so the tests are skipped rather than failed.
 */
public class Lucene99ScorerPatcherTests extends KNNTestCase {

    private PrefetchableScorerTestUtils.SharedScorerState scorerState;

    @Before
    @SneakyThrows
    public void resolveSharedScorerAndReset() {
        scorerState = PrefetchableScorerTestUtils.resolveAndReset();
    }

    @After
    @SneakyThrows
    public void restoreGlobalState() {
        if (scorerState != null) {
            scorerState.restore();
        }
    }

    /**
     * End-to-end check that the global patch is actually visible through a freshly-initialized
     * {@link Lucene99HnswVectorsFormat} (not just the raw static field): the format shares the single patched
     * {@code Lucene99FlatVectorsFormat}, and its {@code toString()} embeds the scorer, so after {@code installOnce()}
     * every new format instance reports the {@link PrefetchableFlatVectorScorer}.
     */
    @Test
    @SneakyThrows
    public void testFreshFormatInstanceReflectsGlobalPatch() {
        final String prefetchableMarker = PrefetchableFlatVectorScorer.class.getSimpleName();

        // Before patching, a newly constructed format must expose the stock (non-prefetchable) scorer.
        assertFalse(
            "precondition: a fresh Lucene99HnswVectorsFormat must not report a prefetchable scorer before installOnce()",
            new Lucene99HnswVectorsFormat().toString().contains(prefetchableMarker)
        );

        Lucene99ScorerPatcher.installOnce();

        // After patching, any newly constructed format must report the prefetchable scorer through its public API.
        assertTrue(
            "a Lucene99HnswVectorsFormat initialized after installOnce() must report the prefetchable scorer",
            new Lucene99HnswVectorsFormat().toString().contains(prefetchableMarker)
        );
    }
}
