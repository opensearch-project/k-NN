/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.search.IndexSearcher;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptFactory;
import org.opensearch.search.lookup.SearchLookup;

import java.util.Map;

public class KNNScoreScriptFactory implements ScoreScript.Factory, ScriptFactory {
    @Override
    public boolean isResultDeterministic() {
        // This implies the results are cacheable
        return true;
    }

    @Override
    public ScoreScript.LeafFactory newFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher indexSearcher) {
        return new KNNScoreScriptLeafFactory(params, lookup, indexSearcher);
    }
}
