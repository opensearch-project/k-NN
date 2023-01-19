/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.Builder;
import lombok.Value;

@Value
@Builder
public class NativeMemoryCacheManagerDto {
    boolean isWeightLimited;
    long maxWeight;
    boolean isExpirationLimited;
    long expiryTimeInMin;
}
