/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

/*
 * A factory that creates KNNScoreScriptLeafFactory objects. The factory is responsible for parsing the parameters
 * passed in the query and creating the KNNScoreScriptLeafFactory object.
 */
public class KNNScoreScriptLeafFactory implements ScoreScript.LeafFactory {
    private final Map<String, Object> params;
    private final SearchLookup lookup;
    private final String similaritySpace;
    private final String field;
    private final Object query;
    private final KNNScoringSpace knnScoringSpace;
    private final IndexSearcher searcher;

    public KNNScoreScriptLeafFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
        KNNCounter.SCRIPT_QUERY_REQUESTS.increment();
        this.params = params;
        this.lookup = lookup;
        this.field = getValue(params, "field").toString();
        this.similaritySpace = getValue(params, "space_type").toString();
        this.query = getValue(params, "query_value");
        this.searcher = searcher;

        this.knnScoringSpace = KNNScoringSpaceFactory.create(
            this.similaritySpace,
            this.query,
            lookup.doc().mapperService().fieldType(this.field)
        );
    }

    private Object getValue(Map<String, Object> params, String fieldName) {
        final Object value = params.get(fieldName);
        if (value != null) return value;

        KNNCounter.SCRIPT_QUERY_ERRORS.increment();
        throw new IllegalArgumentException(String.format(Locale.ROOT, "Missing parameter [%s]", fieldName));
    }

    @Override
    public boolean needs_score() {
        return false;
    }

    /**
     * For each segment, supply the KNNScoreScript that should be used to re-score the documents returned from the
     * query. Because the method to score the documents was set during factory construction, the scripts are agnostic of
     * the similarity space. The KNNScoringSpace will return the correct script, given the query, the field type, and
     * the similarity space.
     *
     * @param ctx LeafReaderContext for the segment
     * @return ScoreScript to be executed
     */
    @Override
    public ScoreScript newInstance(LeafReaderContext ctx) throws IOException {
        return knnScoringSpace.getScoreScript(params, field, lookup, ctx, this.searcher);
    }
}
