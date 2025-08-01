/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptFactory;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.Map;

/**
 * Enhanced KNN Score Script Factory that supports multiple script types:
 * 1. Original parameter-based approach: knn_score with field, space_type, query_value parameters
 * 2. Function syntax: cosineSimilarity(params.query, doc['field'])
 * 3. Mustache templates: {{#condition}}cosineSimilarity(...){{/condition}}
 */
public class KNNScoreScriptFactory implements ScoreScript.Factory, ScriptFactory {

    private final String scriptSource;

    /**
     * Constructor for original knn_score approach (no script source)
     */
    public KNNScoreScriptFactory() {
        this.scriptSource = null;
    }

    /**
     * Constructor for function/Mustache syntax (with script source)
     */
    public KNNScoreScriptFactory(String scriptSource) {
        // Validate that script contains either k-NN functions or Mustache templates
        boolean hasKNNFunction = KNNMustacheScript.containsKNNFunctions(scriptSource);
        boolean isMustacheTemplate = KNNMustacheScript.isMustacheTemplate(scriptSource);

        if (!hasKNNFunction && !isMustacheTemplate) {
            throw new IllegalArgumentException(
                "Script must contain either k-NN functions or Mustache template syntax. Got: " + scriptSource
            );
        }

        this.scriptSource = scriptSource;
    }

    @Override
    public boolean isResultDeterministic() {
        // This implies the results are cacheable
        return true;
    }

    @Override
    public ScoreScript.LeafFactory newFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher indexSearcher) {
        if (scriptSource == null) {
            // Original parameter-based approach
            return new KNNScoreScriptLeafFactory(params, lookup, indexSearcher);
        } else {
            // New function/Mustache syntax approach
            return new ScoreScript.LeafFactory() {
                @Override
                public boolean needs_score() {
                    return false;
                }

                @Override
                public ScoreScript newInstance(LeafReaderContext context) throws IOException {
                    return new KNNMustacheScript(scriptSource, params, lookup, context);
                }
            };
        }
    }
}
