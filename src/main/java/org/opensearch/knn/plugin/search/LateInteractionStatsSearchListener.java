/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.search;

import org.apache.lucene.search.BooleanClause;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilderVisitor;
import org.opensearch.index.query.ScriptQueryBuilder;
import org.opensearch.index.query.functionscore.FunctionScoreQueryBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreFunctionBuilder;
import org.opensearch.index.query.functionscore.ScriptScoreQueryBuilder;
import org.opensearch.index.shard.SearchOperationListener;
import org.opensearch.knn.plugin.stats.KNNCounter;
import org.opensearch.script.Script;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.rescore.QueryRescorerBuilder;
import org.opensearch.search.rescore.RescorerBuilder;
import org.opensearch.search.sort.ScriptSortBuilder;
import org.opensearch.search.sort.SortBuilder;

import java.io.IOException;
import java.util.Map;

/**
 * Tracks domain adoption of the late interaction scoring function.
 *
 * <p>The {@code lateInteractionScore} Painless function is a static, per-document
 * allowlisted method, so incrementing inside it counts once per scored document
 * (a single request scoring N docs would count N times). To measure adoption at
 * the request level, this listener instead increments the counter once per
 * shard-level query phase whose source uses the function. This matches the
 * granularity of the other k-NN query counters (e.g. {@code knn_query_requests}),
 * which are also node/shard-level, and it does not touch the Painless API.
 *
 * <p>Only script-carrying components are inspected: {@code script_score},
 * {@code script} queries, {@code function_score} script functions (walked through
 * the query tree with a {@link QueryBuilderVisitor}), query rescorers, script fields
 * and script sorts. Only the script source is matched, so a field name or query text
 * containing the marker does not count, and requests without scripts are never
 * serialized.
 */
public class LateInteractionStatsSearchListener implements SearchOperationListener {

    static final String LATE_INTERACTION_MARKER = "lateInteractionScore";

    @Override
    public void onPreQueryPhase(SearchContext searchContext) {
        if (searchContext == null || searchContext.request() == null) {
            return;
        }
        if (usesLateInteraction(searchContext.request().source())) {
            KNNCounter.LATE_INTERACTION_SCORE_REQUESTS.increment();
        }
    }

    /**
     * Returns true when any script in the search source calls the late interaction function.
     */
    static boolean usesLateInteraction(SearchSourceBuilder source) {
        if (source == null) {
            return false;
        }
        if (queryUsesLateInteraction(source.query()) || queryUsesLateInteraction(source.postFilter())) {
            return true;
        }
        if (source.rescores() != null) {
            for (RescorerBuilder<?> rescorer : source.rescores()) {
                if (rescorer instanceof QueryRescorerBuilder
                    && queryUsesLateInteraction(((QueryRescorerBuilder) rescorer).getRescoreQuery())) {
                    return true;
                }
            }
        }
        if (source.scriptFields() != null) {
            for (SearchSourceBuilder.ScriptField scriptField : source.scriptFields()) {
                if (scriptUsesLateInteraction(scriptField.script())) {
                    return true;
                }
            }
        }
        if (source.sorts() != null) {
            for (SortBuilder<?> sort : source.sorts()) {
                if (sort instanceof ScriptSortBuilder && scriptUsesLateInteraction(((ScriptSortBuilder) sort).script())) {
                    return true;
                }
            }
        }
        return false;
    }

    static boolean queryUsesLateInteraction(QueryBuilder query) {
        if (query == null) {
            return false;
        }
        LateInteractionVisitor visitor = new LateInteractionVisitor();
        query.visit(visitor);
        return visitor.found;
    }

    static boolean scriptUsesLateInteraction(Script script) {
        return script != null && script.getIdOrCode() != null && script.getIdOrCode().contains(LATE_INTERACTION_MARKER);
    }

    /**
     * Visits the query tree and checks only script-carrying nodes.
     */
    private static final class LateInteractionVisitor implements QueryBuilderVisitor {
        private boolean found;

        @Override
        public void accept(QueryBuilder qb) {
            if (found || qb == null) {
                return;
            }
            if (qb instanceof ScriptScoreQueryBuilder) {
                ScriptScoreQueryBuilder scriptScore = (ScriptScoreQueryBuilder) qb;
                found = scriptUsesLateInteraction(scriptOf(scriptScore));
                // ScriptScoreQueryBuilder#visit only accepts its inner query without descending into it,
                // so walk the inner query explicitly to reach nested script carriers.
                if (found == false && scriptScore.query() != null) {
                    scriptScore.query().visit(this);
                }
            } else if (qb instanceof ScriptQueryBuilder) {
                found = scriptUsesLateInteraction(((ScriptQueryBuilder) qb).script());
            } else if (qb instanceof FunctionScoreQueryBuilder) {
                for (FunctionScoreQueryBuilder.FilterFunctionBuilder function : ((FunctionScoreQueryBuilder) qb).filterFunctionBuilders()) {
                    if (function.getScoreFunction() instanceof ScriptScoreFunctionBuilder
                        && scriptUsesLateInteraction(((ScriptScoreFunctionBuilder) function.getScoreFunction()).getScript())) {
                        found = true;
                        return;
                    }
                }
            }
        }

        @Override
        public QueryBuilderVisitor getChildVisitor(BooleanClause.Occur occur) {
            return this;
        }
    }

    /**
     * ScriptScoreQueryBuilder does not expose its script, so read it back from this node's own
     * XContent. This only runs for script_score nodes, never for non-script queries.
     */
    @SuppressWarnings("unchecked")
    private static Script scriptOf(ScriptScoreQueryBuilder scriptScore) {
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            scriptScore.toXContent(builder, ToXContent.EMPTY_PARAMS);
            Map<String, Object> node = XContentHelper.convertToMap(BytesReference.bytes(builder), false, MediaTypeRegistry.JSON).v2();
            Object body = node.get(ScriptScoreQueryBuilder.NAME);
            if (body instanceof Map == false) {
                return null;
            }
            Object script = ((Map<String, Object>) body).get(ScriptScoreQueryBuilder.SCRIPT_FIELD.getPreferredName());
            if (script == null) {
                return null;
            }
            return Script.parse(script);
        } catch (IOException | RuntimeException e) {
            // Stats tracking must never fail a search.
            return null;
        }
    }
}
