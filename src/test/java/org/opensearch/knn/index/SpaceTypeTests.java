/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.KNNTestCase;

public class SpaceTypeTests extends KNNTestCase {

    public void testSpaceTypeToVectorSimilarityFunction_l2() {
        assertEquals(VectorSimilarityFunction.EUCLIDEAN, SpaceType.spaceTypeToVectorSimilarityFunction(SpaceType.L2));
    }

    public void testSpaceTypeToVectorSimilarityFunction_invalid() {
        expectThrows(IllegalArgumentException.class, () -> SpaceType.spaceTypeToVectorSimilarityFunction(SpaceType.L1));
    }
}
