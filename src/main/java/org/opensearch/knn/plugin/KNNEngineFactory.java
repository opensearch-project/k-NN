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
 *   Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


package org.opensearch.knn.plugin;

import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.InternalEngine;

/**
 * EngineFactory to inject the KNNCodecService to help segments write using the KNNCodec.
 */
class KNNEngineFactory implements EngineFactory {

    private static KNNCodecService codecService = new KNNCodecService();

    @Override
    public Engine newReadWriteEngine(EngineConfig config) {
        codecService.setPostingsFormat(config.getCodec().postingsFormat());
        EngineConfig engineConfig = new EngineConfig(config.getShardId(),
                config.getThreadPool(), config.getIndexSettings(), config.getWarmer(), config.getStore(),
                config.getMergePolicy(), config.getAnalyzer(), config.getSimilarity(), codecService,
                config.getEventListener(), config.getQueryCache(), config.getQueryCachingPolicy(),
                config.getTranslogConfig(), config.getFlushMergesAfter(), config.getExternalRefreshListener(),
                config.getInternalRefreshListener(), config.getIndexSort(), config.getCircuitBreakerService(),
                config.getGlobalCheckpointSupplier(), config.retentionLeasesSupplier(), config.getPrimaryTermSupplier(),
                config.getTombstoneDocSupplier());
        return new InternalEngine(engineConfig);
    }
}
