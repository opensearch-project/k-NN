/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.rescore;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class RescoreContext {

    public static final float DEFAULT_OVERSAMPLE_FACTOR = 1.0f;
    public static final float MAX_OVERSAMPLE_FACTOR = 100.0f;
    public static final float MIN_OVERSAMPLE_FACTOR = 0.0f;

    @Builder.Default
    private float oversampleFactor = DEFAULT_OVERSAMPLE_FACTOR;

    /**
     *
     * @return default RescoreContext
     */
    public static RescoreContext getDefault() {
        return RescoreContext.builder().build();
    }
}
