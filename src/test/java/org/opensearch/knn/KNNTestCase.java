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

package org.opensearch.knn;

import org.opensearch.knn.index.KNNIndexCache;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.test.OpenSearchTestCase;

/**
 * Base class for integration tests for KNN plugin. Contains several methods for testing KNN ES functionality.
 */
public class KNNTestCase extends OpenSearchTestCase {
    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        resetState();
    }

    public static void resetState() {
        // Reset all of the counters
        for (KNNCounter knnCounter : KNNCounter.values()) {
            knnCounter.set(0L);
        }

        // Clean up the cache
        KNNIndexCache.getInstance().evictAllGraphsFromCache();
        KNNIndexCache.getInstance().close();
    }
}