/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.Builder;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.knn.index.mapper.CompressionLevel;

/**
 * Small data class for storing info that gets resolved during resolution process
 */
@RequiredArgsConstructor
@Getter
@Builder
public class ResolvedMethodContext {
    private final KNNMethodContext knnMethodContext;
    @Builder.Default
    private final CompressionLevel compressionLevel = CompressionLevel.NOT_CONFIGURED;
}
