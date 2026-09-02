/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.test.OpenSearchTestCase;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * The method-level behavioral overrides a runtime-registered engine can set when building its own spec.
 * Unset overrides must leave every core-derived answer unchanged.
 */
public class ResolvedIndexSpecOverrideTests extends OpenSearchTestCase {

    private static ResolvedIndexSpec.ResolvedIndexSpecBuilder faissHnswFlat() {
        return ResolvedIndexSpec.builder()
            .engine(KNNEngine.FAISS)
            .methodName(METHOD_HNSW)
            .encoderType(Encoder.EncoderType.FLAT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .indexVersionCreated(Version.CURRENT);
    }

    public void testUnsetOverridesLeaveCoreAnswersUnchanged() {
        final ResolvedIndexSpec withNulls = faissHnswFlat().rescoreDefaultOverride(null).memoryOptimizedEligibleOverride(null).build();
        final ResolvedIndexSpec plain = faissHnswFlat().build();
        assertEquals(plain.isMemoryOptimizedEligible(), withNulls.isMemoryOptimizedEligible());
        assertEquals(plain.getRescoreContext(), withNulls.getRescoreContext());
        assertTrue(plain.isMemoryOptimizedEligible());
    }

    public void testMemoryOptimizedEligibleOverrideAnswersBothWays() {
        assertFalse(faissHnswFlat().memoryOptimizedEligibleOverride(false).build().isMemoryOptimizedEligible());
        // An engine that is not eligible by core rules becomes eligible only through the override.
        final ResolvedIndexSpec.ResolvedIndexSpecBuilder nonEligible = faissHnswFlat().engine(KNNEngine.LUCENE);
        assertFalse(nonEligible.build().isMemoryOptimizedEligible());
        assertTrue(nonEligible.memoryOptimizedEligibleOverride(true).build().isMemoryOptimizedEligible());
    }

    public void testRescoreDefaultOverrideWinsAndImpliesRequiresRescore() {
        final RescoreContext override = RescoreContext.builder().oversampleFactor(3.0f).userProvided(false).build();
        final ResolvedIndexSpec spec = faissHnswFlat().rescoreDefaultOverride(override).build();
        assertSame(override, spec.getRescoreContext());
        assertTrue(spec.requiresRescore());
        // Without the override this configuration has no default rescore.
        assertNull(faissHnswFlat().build().getRescoreContext());
        assertFalse(faissHnswFlat().build().requiresRescore());
    }
}
