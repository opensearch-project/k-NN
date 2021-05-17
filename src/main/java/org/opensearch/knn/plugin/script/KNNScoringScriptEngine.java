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

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;

import java.util.Map;
import java.util.Set;

/**
 * KNN Custom scoring Engine implementation.
 */
public class KNNScoringScriptEngine implements ScriptEngine {

    public static final String NAME = "knn";
    public static final String SCRIPT_SOURCE = "knn_score";

    @Override
    public String getType() {
        return NAME;
    }

    @Override
    public <FactoryType> FactoryType compile(String name, String code, ScriptContext<FactoryType> context, Map<String, String> params) {
        KNNCounter.SCRIPT_COMPILATIONS.increment();
        if (!ScoreScript.CONTEXT.equals(context)) {
            KNNCounter.SCRIPT_COMPILATION_ERRORS.increment();
            throw new IllegalArgumentException(getType() + " KNN scoring scripts cannot be used for context ["
                    + context.name + "]");
        }
        // we use the script "source" as the script identifier
        if (!SCRIPT_SOURCE.equals(code)) {
            KNNCounter.SCRIPT_COMPILATION_ERRORS.increment();
            throw new IllegalArgumentException("Unknown script name " + code);
        }
        ScoreScript.Factory factory = KNNScoreScriptFactory::new;
        return context.factoryClazz.cast(factory);
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return null;
    }
}
