/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.util.Arrays;
import java.util.Locale;

public enum SearchMode {
    ANN("ann") {

    },
    EXACT("exact") {

    };

    private final String value;

    SearchMode(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static SearchMode getSearchMode(String searchModeName) {
        for (SearchMode currentSearchMode : SearchMode.values()) {
            if (currentSearchMode.getValue().equalsIgnoreCase(searchModeName)) {
                return currentSearchMode;
            }
        }
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "Unable to find search mode: %s . Valid values are: %s",
                searchModeName,
                Arrays.toString(SearchMode.values())
            )
        );
    }
}
