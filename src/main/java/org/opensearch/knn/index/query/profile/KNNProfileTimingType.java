/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.profile;

import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

public enum KNNProfileTimingType {

    EXACT_KNN_SEARCH,
    ANN_SEARCH;

    public static Set<String> getAllValues() {
        return Arrays.stream(KNNProfileTimingType.values()).map(Enum::name).collect(Collectors.toSet());
    }
}
