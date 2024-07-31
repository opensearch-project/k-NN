/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Class stores information about the shared memory allocations between loaded native indices.
 */
@RequiredArgsConstructor
@Getter
public class SharedIndexState {
    private final long sharedIndexStateAddress;
    private final String modelId;
    private final KNNEngine knnEngine;
}
