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
/*
 *   Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */


package org.opensearch.knn.plugin.script;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.index.mapper.MappedFieldType;

/**
 * Factory to create correct KNNScoringSpace based on the spaceType passed in.
 */
public class KNNScoringSpaceFactory {
    public static KNNScoringSpace create(String spaceType, Object query, MappedFieldType mappedFieldType) {
        if (KNNConstants.HAMMING_BIT.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.HammingBit(query, mappedFieldType);
        }

        if (KNNConstants.L2.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.L2(query, mappedFieldType);
        }
        if (KNNConstants.L1.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.L1(query, mappedFieldType);
        }
        if (KNNConstants.LINF.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.LInf(query, mappedFieldType);
        }

        if (KNNConstants.INNER_PROD.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.InnerProd(query, mappedFieldType);
        }

        if (KNNConstants.COSINESIMIL.equalsIgnoreCase(spaceType)) {
            return new KNNScoringSpace.CosineSimilarity(query, mappedFieldType);
        }

        KNNCounter.SCRIPT_QUERY_ERRORS.increment();
        throw new IllegalArgumentException("Invalid space type. Please refer to the available space types.");
    }
}
