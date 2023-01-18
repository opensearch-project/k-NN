/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class NativeMemoryCacheManagerDto {
    private final boolean isWeightLimited;
    private final long maxWeight;
    private final boolean isExpirationLimited;
    private final long expiryTimeInMin;
}
