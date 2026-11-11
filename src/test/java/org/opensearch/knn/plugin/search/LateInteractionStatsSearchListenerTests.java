/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.search;

import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.MatchQueryBuilder;
import org.opensearch.index.query.ScriptQueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.query.functionscore.FunctionScoreQueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreFunctionBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptType;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.internal.ShardSearchRequest;
import org.opensearch.search.rescore.QueryRescorerBuilder;
import org.opensearch.search.sort.ScriptSortBuilder;

import java.util.Collections;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class LateInteractionStatsSearchListenerTests extends KNNTestCase {

    private static final String LI_SOURCE = "lateInteractionScore(params.query_vector, 'my_vector', params._source)";
    private static final String OTHER_SOURCE = "doc['price'].value * 2";

    private static Script script(String source) {
        return new Script(ScriptType.INLINE, Script.DEFAULT_SCRIPT_LANG, source, Collections.emptyMap());
    }

    public void testScriptScoreQuery_counts() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(
            new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script(LI_SOURCE))
        );
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    public void testScriptScoreQuery_otherScript_doesNotCount() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(
            new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script(OTHER_SOURCE))
        );
        assertFalse(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    public void testScriptScoreNestedInBool_counts() {
        BoolQueryBuilder bool = new BoolQueryBuilder().must(new TermQueryBuilder("tag", "a"))
            .should(new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script(LI_SOURCE)));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(new SearchSourceBuilder().query(bool)));
    }

    public void testScriptScoreNestedInScriptScore_counts() {
        ScriptScoreQueryBuilder inner = new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script(LI_SOURCE));
        ScriptScoreQueryBuilder outer = new ScriptScoreQueryBuilder(new BoolQueryBuilder().must(inner), script(OTHER_SOURCE));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(new SearchSourceBuilder().query(outer)));
    }

    public void testScriptQuery_counts() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(new ScriptQueryBuilder(script(LI_SOURCE)));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    public void testFunctionScoreScriptFunction_counts() {
        FunctionScoreQueryBuilder functionScore = new FunctionScoreQueryBuilder(
            new MatchAllQueryBuilder(),
            new ScriptScoreFunctionBuilder(script(LI_SOURCE))
        );
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(new SearchSourceBuilder().query(functionScore)));
    }

    public void testRescoreQuery_counts() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(new MatchAllQueryBuilder())
            .addRescorer(new QueryRescorerBuilder(new ScriptScoreQueryBuilder(new MatchAllQueryBuilder(), script(LI_SOURCE))));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    public void testScriptField_counts() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(new MatchAllQueryBuilder()).scriptField("li", script(LI_SOURCE));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    public void testScriptSort_counts() {
        SearchSourceBuilder source = new SearchSourceBuilder().query(new MatchAllQueryBuilder())
            .sort(new ScriptSortBuilder(script(LI_SOURCE), ScriptSortBuilder.ScriptSortType.NUMBER));
        assertTrue(LateInteractionStatsSearchListener.usesLateInteraction(source));
    }

    /**
     * The marker appearing outside a script (field name or query text) must not count, which is the
     * false positive the earlier toString()-based check had.
     */
    public void testMarkerOutsideScript_doesNotCount() {
        BoolQueryBuilder bool = new BoolQueryBuilder().must(new MatchQueryBuilder("description", "lateInteractionScore"))
            .filter(new TermQueryBuilder("lateInteractionScore", "x"));
        assertFalse(LateInteractionStatsSearchListener.usesLateInteraction(new SearchSourceBuilder().query(bool)));
    }

    public void testNoQuery_doesNotCount() {
        assertFalse(LateInteractionStatsSearchListener.usesLateInteraction(null));
        assertFalse(LateInteractionStatsSearchListener.usesLateInteraction(new SearchSourceBuilder()));
    }

    public void testOnPreQueryPhase_incrementsOnlyForLateInteraction() {
        LateInteractionStatsSearchListener listener = new LateInteractionStatsSearchListener();
        long before = KNNCounter.LATE_INTERACTION_SCORE_REQUESTS.getCount();

        listener.onPreQueryPhase(contextWith(new SearchSourceBuilder().query(new ScriptQueryBuilder(script(LI_SOURCE)))));
        assertEquals(before + 1, (long) KNNCounter.LATE_INTERACTION_SCORE_REQUESTS.getCount());

        listener.onPreQueryPhase(contextWith(new SearchSourceBuilder().query(new MatchAllQueryBuilder())));
        listener.onPreQueryPhase(contextWith(null));
        listener.onPreQueryPhase(null);
        assertEquals(before + 1, (long) KNNCounter.LATE_INTERACTION_SCORE_REQUESTS.getCount());
    }

    private static SearchContext contextWith(SearchSourceBuilder source) {
        SearchContext context = mock(SearchContext.class);
        ShardSearchRequest request = mock(ShardSearchRequest.class);
        when(request.source()).thenReturn(source);
        when(context.request()).thenReturn(request);
        return context;
    }
}
