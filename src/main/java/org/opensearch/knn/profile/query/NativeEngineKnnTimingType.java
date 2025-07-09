/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.profile.query;

import java.util.Locale;

public enum NativeEngineKnnTimingType {
    EXPAND_NESTED_DOCS,
    RESCORE;

    @Override
    public String toString() {
        return name().toLowerCase(Locale.ROOT);
    }
}
