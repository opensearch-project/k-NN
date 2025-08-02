/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;

import java.util.Collections;
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
            throw new IllegalArgumentException(getType() + " KNN scoring scripts cannot be used for context [" + context.name + "]");
        }

        ScoreScript.Factory factory;

        // Support original knn_score, function syntax, and Mustache templates
        if (SCRIPT_SOURCE.equals(code)) {
            // Original knn_score syntax (maintain backward compatibility)
            factory = new KNNScoreScriptFactory();
        } else if (KNNMustacheScript.isMustacheTemplate(code) || KNNMustacheScript.containsKNNFunctions(code)) {
            // Enhanced KNNScoreScriptFactory handles both Mustache templates and direct k-NN function calls
            factory = new KNNScoreScriptFactory(code);
        } else {
            KNNCounter.SCRIPT_COMPILATION_ERRORS.increment();
            throw new IllegalArgumentException(
                "Script must be 'knn_score', contain k-NN functions like cosineSimilarity(params.query, doc['field']), or use Mustache templates"
            );
        }

        return context.factoryClazz.cast(factory);
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return Collections.singleton(ScoreScript.CONTEXT);
    }
}
