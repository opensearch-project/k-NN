/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.index.mapper.MappedFieldType;

/**
 * Factory to create correct KNNScoringSpace based on the spaceType passed in.
 */
public class KNNScoringSpaceFactory {
    public static final String HAMMING_BIT = "hammingbit";

    public static KNNScoringSpace create(String spaceType, Object query, MappedFieldType mappedFieldType) {
        if (SpaceType.HAMMING.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.Hamming(query, mappedFieldType);
        }

        if (SpaceType.L2.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.L2(query, mappedFieldType);
        }
        if (SpaceType.L1.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.L1(query, mappedFieldType);
        }
        if (SpaceType.LINF.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.LInf(query, mappedFieldType);
        }

        if (SpaceType.INNER_PRODUCT.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.InnerProd(query, mappedFieldType);
        }

        if (SpaceType.COSINESIMIL.getValue().equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.CosineSimilarity(query, mappedFieldType);
        }

        if (HAMMING_BIT.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.HammingBit(query, mappedFieldType);
        }

        KNNCounter.SCRIPT_QUERY_ERRORS.increment();
        throw new IllegalArgumentException("Invalid space type. Please refer to the available space types.");
    }
}
