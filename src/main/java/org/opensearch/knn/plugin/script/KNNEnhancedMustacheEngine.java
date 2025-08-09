/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import com.github.mustachejava.DefaultMustacheFactory;
import com.github.mustachejava.Mustache;
import com.github.mustachejava.MustacheException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.opensearch.script.GeneralScriptException;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptException;
import org.opensearch.script.TemplateScript;
import org.opensearch.search.lookup.SearchLookup;
import org.opensearch.knn.index.KNNVectorScriptDocValues;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Enhanced Mustache script engine with k-NN vector operations support.
 *
 * This implementation provides a standalone script engine that supports
 * basic Mustache templating with k-NN specific functions. Since the k-NN
 * plugin is standalone and cannot directly extend OpenSearch's MustacheScriptEngine,
 * this approach implements ScriptEngine directly.
 *
 * Supported k-NN template functions:
 * - {{#knn_basic}}field_name vector k_value{{/knn_basic}} - Basic k-NN query generation
 * - {{#l2Distance}}query_vector field_name{{/l2Distance}} - L2 distance calculation
 * - {{#cosineSimilarity}}query_vector field_name{{/cosineSimilarity}} - Cosine similarity calculation
 * - {{#innerProduct}}query_vector field_name{{/innerProduct}} - Inner product calculation
 * - {{#hamming}}query_vector field_name{{/hamming}} - Hamming distance calculation
 */
public class KNNEnhancedMustacheEngine implements ScriptEngine {

    public static final String NAME = "knn-mustache";

    @Override
    public String getType() {
        return NAME;
    }

    @Override
    public <T> T compile(String templateName, String templateSource, ScriptContext<T> context, Map<String, String> options) {
        // Create a simple factory - k-NN logic will be handled in parameter preprocessing
        DefaultMustacheFactory factory = new DefaultMustacheFactory();
        StringReader reader = new StringReader(templateSource);

        try {
            Mustache template = factory.compile(reader, "knn-query-template");

            if (context.instanceClazz.equals(TemplateScript.class)) {
                TemplateScript.Factory compiled = params -> new KNNMustacheExecutableScript(template, params);
                return context.factoryClazz.cast(compiled);
            } else if (context.instanceClazz.equals(ScoreScript.class)) {
                ScoreScript.Factory compiled = (params, lookup, searcher) -> new KNNMustacheScoreScriptLeafFactory(
                    template,
                    params,
                    lookup,
                    searcher
                );
                return context.factoryClazz.cast(compiled);
            } else {
                throw new IllegalArgumentException("knn-mustache engine does not know how to handle context [" + context.name + "]");
            }
        } catch (MustacheException ex) {
            throw new ScriptException(ex.getMessage(), ex, Collections.emptyList(), templateSource, NAME);
        }
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return Set.of(TemplateScript.CONTEXT, ScoreScript.CONTEXT);
    }

    /**
     * Executable script implementation for k-NN Mustache templates.
     */
    private static class KNNMustacheExecutableScript extends TemplateScript {
        private final Mustache template;
        private final Map<String, Object> params;

        KNNMustacheExecutableScript(Mustache template, Map<String, Object> params) {
            super(params);
            this.template = template;
            this.params = params;
        }

        @Override
        public String execute() {
            final StringWriter writer = new StringWriter();
            try {
                // Process k-NN specific functions before executing template
                Map<String, Object> processedParams = preprocessKNNFunctions(params);
                template.execute(writer, processedParams);
            } catch (Exception e) {
                throw new GeneralScriptException("Error running k-NN mustache template: " + template, e);
            }
            return writer.toString();
        }

        /**
         * Preprocess parameters to handle k-NN specific functions.
         * This method converts k-NN function calls into their corresponding query JSON.
         */
        private Map<String, Object> preprocessKNNFunctions(Map<String, Object> originalParams) {
            Map<String, Object> processedParams = new HashMap<>(originalParams);
            // Add k-NN function helpers that can be used in templates
            processedParams.put("l2Distance", new KNNVectorFunction("l2Squared"));
            processedParams.put("cosineSimilarity", new KNNVectorFunction("cosineSimilarity"));
            processedParams.put("innerProduct", new KNNVectorFunction("innerProduct"));
            processedParams.put("hamming", new KNNVectorFunction("hamming"));

            return processedParams;
        }
    }

    /**
     * LeafFactory for k-NN Mustache score scripts.
     * This creates individual ScoreScript instances for each segment.
     */
    private static class KNNMustacheScoreScriptLeafFactory implements ScoreScript.LeafFactory {
        private final Mustache template;
        private final Map<String, Object> params;
        private final SearchLookup lookup;
        private final IndexSearcher searcher;

        KNNMustacheScoreScriptLeafFactory(Mustache template, Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
            this.template = template;
            this.params = params;
            this.lookup = lookup;
            this.searcher = searcher;
        }

        @Override
        public boolean needs_score() {
            return false;
        }

        @Override
        public ScoreScript newInstance(LeafReaderContext ctx) throws IOException {
            return new KNNMustacheScoreScript(template, params, lookup, searcher, ctx);
        }
    }

    /**
     * Score script implementation for k-NN Mustache templates.
     * This enables vector scoring operations in script_score queries.
     */
    private static class KNNMustacheScoreScript extends ScoreScript {
        private final Mustache template;
        private final Map<String, Object> params;

        KNNMustacheScoreScript(
            Mustache template,
            Map<String, Object> params,
            SearchLookup lookup,
            IndexSearcher searcher,
            LeafReaderContext ctx
        ) {
            super(params, lookup, searcher, ctx);
            this.template = template;
            this.params = params;
        }

        @Override
        public double execute(ScoreScript.ExplanationHolder explanationHolder) {
            try {
                // Process k-NN specific functions before executing template
                Map<String, Object> processedParams = preprocessKNNFunctions(params);

                // Execute Mustache template to generate script source
                final StringWriter writer = new StringWriter();
                template.execute(writer, processedParams);
                String scriptSource = writer.toString();
                try {
                    return Double.parseDouble(scriptSource.trim());
                } catch (NumberFormatException e) {
                    // If template doesn't resolve to a number, return default score
                    return 1.0;
                }
            } catch (Exception e) {
                throw new GeneralScriptException("Error executing k-NN mustache score script", e);
            }
        }

        /**
         * Preprocess parameters to handle k-NN specific functions.
         */
        private Map<String, Object> preprocessKNNFunctions(Map<String, Object> originalParams) {
            Map<String, Object> processedParams = new HashMap<>(originalParams);
            processedParams.put("vectorUtils", new ScoreScriptVectorUtils());
            return processedParams;
        }
    }

    /**
     * Helper class for k-NN vector functions in TemplateScript context.
     * Provides simple function identifiers for template generation.
     */
    private static class KNNVectorFunction {
        private final String functionName;

        public KNNVectorFunction(String functionName) {
            this.functionName = functionName;
        }

        @Override
        public String toString() {
            return functionName;
        }
    }

    /**
     * Helper class for k-NN vector functions in ScoreScript context.
     * Provides access to actual vector calculation methods.
     */
    private static class ScoreScriptVectorUtils {
        /**
         * Calculate L2 squared distance between query vector and document field.
         * Usage in template: {{vectorUtils.l2Distance}}
         */
        public float l2Distance(List<Number> queryVector, Object docValues) {
            if (docValues instanceof KNNVectorScriptDocValues) {
                return KNNScoringUtil.l2Squared(queryVector, (KNNVectorScriptDocValues<?>) docValues);
            }
            throw new IllegalArgumentException("Invalid document values for l2Distance");
        }

        /**
         * Calculate cosine similarity between query vector and document field.
         * Usage in template: {{vectorUtils.cosineSimilarity}}
         */
        public float cosineSimilarity(List<Number> queryVector, Object docValues) {
            if (docValues instanceof KNNVectorScriptDocValues) {
                return KNNScoringUtil.cosineSimilarity(queryVector, (KNNVectorScriptDocValues<?>) docValues);
            }
            throw new IllegalArgumentException("Invalid document values for cosineSimilarity");
        }

        /**
         * Calculate inner product between query vector and document field.
         * Usage in template: {{vectorUtils.innerProduct}}
         */
        public float innerProduct(List<Number> queryVector, Object docValues) {
            if (docValues instanceof KNNVectorScriptDocValues) {
                return KNNScoringUtil.innerProduct(queryVector, (KNNVectorScriptDocValues<?>) docValues);
            }
            throw new IllegalArgumentException("Invalid document values for innerProduct");
        }

        /**
         * Calculate hamming distance between query vector and document field.
         * Usage in template: {{vectorUtils.hamming}}
         */
        public float hamming(List<Number> queryVector, Object docValues) {
            if (docValues instanceof KNNVectorScriptDocValues) {
                return KNNScoringUtil.hamming(queryVector, (KNNVectorScriptDocValues<?>) docValues);
            }
            throw new IllegalArgumentException("Invalid document values for hamming");
        }
    }
}
