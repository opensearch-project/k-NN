/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.script.ScoreScript;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Comprehensive tests for all k-NN Mustache scripting functionality combining unit and integration tests.
 * Tests function parsing, script factories, engine integration, Mustache processing, and real-world examples.
 */
public class KNNMustacheScriptingTests extends OpenSearchTestCase {

    // ========== Function Parser Core Tests ==========

    public void testFunctionParserBasics() {
        // Test function detection
        assertTrue(KNNMustacheScript.containsKNNFunctions("cosineSimilarity(params.query, doc['field'])"));
        assertTrue(KNNMustacheScript.containsKNNFunctions("l2Distance(params.vector, doc['embedding'])"));
        assertTrue(KNNMustacheScript.containsKNNFunctions("cosineSimilarity( params.query , doc['field'] )")); // whitespace
        assertFalse(KNNMustacheScript.containsKNNFunctions("knn_score"));
        assertFalse(KNNMustacheScript.containsKNNFunctions("Math.max(1, 2)"));
        assertFalse(KNNMustacheScript.containsKNNFunctions(""));

        // Test function parsing
        KNNMustacheScript.FunctionCall call = KNNMustacheScript.parseFunction("cosineSimilarity(params.query, doc['field'])");
        assertNotNull(call);
        assertEquals("cosineSimilarity", call.getFunctionName());
        assertEquals("params.query", call.getQueryParam());
        assertEquals("doc['field']", call.getDocField());
        assertEquals("field", call.extractFieldName());
        assertEquals("query", call.extractParamName());

        // Test with double quotes and whitespace
        call = KNNMustacheScript.parseFunction("innerProduct( params.query_vector , doc['field_name'] )");
        assertNotNull(call);
        assertEquals("innerProduct", call.getFunctionName());
        assertEquals("params.query_vector", call.getQueryParam());
        assertEquals("doc['field_name']", call.getDocField());

        // Test all supported functions
        String[] functions = { "cosineSimilarity", "l2Distance", "l2Squared", "innerProduct", "hamming", "l1Norm", "lInfNorm" };
        for (String function : functions) {
            String script = function + "(params.query, doc['field'])";
            assertTrue("Function " + function + " should be recognized", KNNMustacheScript.containsKNNFunctions(script));
            assertNotNull("Function " + function + " should be parseable", KNNMustacheScript.parseFunction(script));
        }

        // Test invalid cases
        assertNull(KNNMustacheScript.parseFunction("knn_score"));
        assertNull(KNNMustacheScript.parseFunction("Math.max(1, 2)"));
        assertNull(KNNMustacheScript.parseFunction(""));
    }

    public void testFunctionParserFieldExtraction() {
        KNNMustacheScript.FunctionCall call = KNNMustacheScript.parseFunction("cosineSimilarity(params.query, doc['my_field'])");
        assertEquals("my_field", call.extractFieldName());
        assertEquals("query", call.extractParamName());

        call = KNNMustacheScript.parseFunction("l2Distance(params.vector, doc[\"embedding_field\"])");
        assertEquals("embedding_field", call.extractFieldName());
        assertEquals("vector", call.extractParamName());

        // Test without quotes
        call = KNNMustacheScript.parseFunction("innerProduct(params.query, doc[field_name])");
        assertEquals("field_name", call.extractFieldName());

        // Test direct field reference
        call = new KNNMustacheScript.FunctionCall("cosineSimilarity", "params.query", "field_name");
        assertEquals("field_name", call.extractFieldName());
    }

    // ========== Script Engine Comprehensive Integration ==========

    public void testScriptEngineAllTypes() {
        KNNScoringScriptEngine engine = new KNNScoringScriptEngine();

        // 1. Original knn_score syntax
        ScoreScript.Factory knnFactory = engine.compile("knn", "knn_score", ScoreScript.CONTEXT, new HashMap<>());
        assertNotNull(knnFactory);
        assertTrue(knnFactory instanceof KNNScoreScriptFactory);

        // 2. Function syntax
        ScoreScript.Factory functionFactory = engine.compile(
            "function",
            "cosineSimilarity(params.query, doc['field'])",
            ScoreScript.CONTEXT,
            new HashMap<>()
        );
        assertNotNull(functionFactory);
        assertTrue(functionFactory instanceof KNNScoreScriptFactory);

        // 3. Mustache syntax
        ScoreScript.Factory mustacheFactory = engine.compile(
            "mustache",
            "{{#condition}}cosineSimilarity(params.query, doc['field']){{/condition}}",
            ScoreScript.CONTEXT,
            new HashMap<>()
        );
        assertNotNull(mustacheFactory);
        assertTrue(mustacheFactory instanceof KNNScoreScriptFactory);

        // All should use the same enhanced KNNScoreScriptFactory (unified approach)
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, knnFactory.getClass());
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, functionFactory.getClass());
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, mustacheFactory.getClass());

        // Test invalid script compilation
        expectThrows(
            IllegalArgumentException.class,
            () -> engine.compile("invalid", "Math.max(1, 2)", ScoreScript.CONTEXT, new HashMap<>())
        );
    }

    // ========== Function Script Factory Validation ==========

    public void testFunctionScriptFactoryComprehensive() {
        // Test valid function scripts
        String[] validScripts = {
            "cosineSimilarity(params.query, doc['field'])",
            "l2Distance(params.vector, doc[\"embedding\"])",
            "innerProduct(params.query_vector, doc['vector_field'])",
            "hamming(params.binary_query, doc['binary_field'])",
            "l1Norm(params.query, doc['field'])",
            "lInfNorm(params.query, doc['field'])",
            "l2Squared(params.query, doc['field'])" };

        for (String script : validScripts) {
            KNNScoreScriptFactory factory = new KNNScoreScriptFactory(script);
            assertNotNull("Factory should be created for: " + script, factory);
            assertTrue(factory.isResultDeterministic());
        }

        // Test invalid scripts
        String[] invalidScripts = {
            "knn_score",
            "Math.max(1, 2)",
            "params.query + doc['field']",
            "",
            "someOtherFunction(params.query, doc['field'])" };
        for (String script : invalidScripts) {
            expectThrows(IllegalArgumentException.class, () -> new KNNScoreScriptFactory(script));
        }
    }

    // ========== Real-World PR Examples ==========

    public void testPRExamplesIntegration() {
        KNNScoringScriptEngine engine = new KNNScoringScriptEngine();

        // PR Example 1: Product similarity
        String productScript = "cosineSimilarity(params.query_vector, doc['product_embedding'])";
        ScoreScript.Factory factory = engine.compile("pr_product", productScript, ScoreScript.CONTEXT, new HashMap<>());
        assertNotNull(factory);
        assertTrue(factory instanceof KNNScoreScriptFactory);

        KNNMustacheScript.FunctionCall call = KNNMustacheScript.parseFunction(productScript);
        assertEquals("query_vector", call.extractParamName());
        assertEquals("product_embedding", call.extractFieldName());

        // PR Example 2: Content similarity
        String contentScript = "l2Distance(params.query_vector, doc['content_embedding'])";
        assertTrue("PR example should be recognized", KNNMustacheScript.containsKNNFunctions(contentScript));
        call = KNNMustacheScript.parseFunction(contentScript);
        assertEquals("l2Distance", call.getFunctionName());
        assertEquals("content_embedding", call.extractFieldName());

        // Test complex real-world field names
        String[] realWorldExamples = {
            "cosineSimilarity(params.search_query, doc['document_vector'])",
            "innerProduct(params.user_preference, doc[\"item_features\"])",
            "l2Distance(params.content_embedding, doc['article_embedding'])" };

        for (String script : realWorldExamples) {
            factory = engine.compile("real_world", script, ScoreScript.CONTEXT, new HashMap<>());
            assertNotNull("Real-world example should compile: " + script, factory);
            call = KNNMustacheScript.parseFunction(script);
            assertTrue("Should extract meaningful field name", call.extractFieldName().length() > 3);
            assertTrue("Should extract meaningful param name", call.extractParamName().length() > 3);
        }
    }

    // ========== Mustache Template Processing ==========

    public void testMustacheProcessorComprehensive() {
        // Template detection
        assertTrue(KNNMustacheScript.isMustacheTemplate("{{variable}}"));
        assertTrue(KNNMustacheScript.isMustacheTemplate("{{#condition}}content{{/condition}}"));
        assertTrue(KNNMustacheScript.isMustacheTemplate("{{^condition}}content{{/condition}}"));
        assertTrue(KNNMustacheScript.isMustacheTemplate("prefix {{var}} suffix"));
        assertFalse(KNNMustacheScript.isMustacheTemplate("cosineSimilarity(params.query, doc['field'])"));
        assertFalse(KNNMustacheScript.isMustacheTemplate("knn_score"));
        assertFalse(KNNMustacheScript.isMustacheTemplate("simple text"));
        assertFalse(KNNMustacheScript.isMustacheTemplate(""));

        Map<String, Object> params = new HashMap<>();
        params.put("has_vector", true);
        params.put("missing_vector", false);
        params.put("value", "test");
        params.put("count", 5);
        params.put("zero", 0);
        params.put("null_value", null);

        // Test variable substitution
        String result = KNNMustacheScript.processTemplate("Hello {{value}}! Number: {{count}} Missing: {{missing}}", params);
        assertEquals("Hello test! Number: 5 Missing: ", result);

        // Test conditional sections
        String template =
            "{{#has_vector}}true{{/has_vector}}{{#missing_vector}}false{{/missing_vector}}{{#count}}count{{/count}}{{#zero}}zero{{/zero}}";
        result = KNNMustacheScript.processTemplate(template, params);
        assertEquals("truecount", result);

        // Test inverted sections
        template =
            "{{^has_vector}}1{{/has_vector}}{{^missing_vector}}2{{/missing_vector}}{{^null_value}}3{{/null_value}}{{^missing_key}}4{{/missing_key}}";
        result = KNNMustacheScript.processTemplate(template, params);
        assertEquals("234", result);

        // Test arrays
        params.put("items", Arrays.asList("a", "b"));
        params.put("empty_list", Collections.emptyList());
        template = "{{#items}}items{{/items}}{{#empty_list}}empty{{/empty_list}}";
        result = KNNMustacheScript.processTemplate(template, params);
        assertEquals("items", result);
    }

    public void testMustacheUserExampleIntegration() {
        // Test the exact user example with full processing flow
        Map<String, Object> params = new HashMap<>();
        String userTemplate =
            "{{#has_vector}}cosineSimilarity(params.query_vector, doc['product_embedding']){{/has_vector}}{{^has_vector}}0{{/has_vector}}";

        // Test Mustache template detection and compilation
        assertTrue("Should detect Mustache template", KNNMustacheScript.isMustacheTemplate(userTemplate));
        KNNScoringScriptEngine engine = new KNNScoringScriptEngine();
        ScoreScript.Factory factory = engine.compile("user_example", userTemplate, ScoreScript.CONTEXT, new HashMap<>());
        assertNotNull("User example should compile", factory);
        assertTrue("Should be Mustache factory", factory instanceof KNNScoreScriptFactory);

        // Test processing logic with has_vector = true
        params.put("has_vector", true);
        String processed = KNNMustacheScript.processTemplate(userTemplate, params);
        assertEquals("cosineSimilarity(params.query_vector, doc['product_embedding'])", processed);
        assertTrue("Processed result should contain k-NN function", KNNMustacheScript.containsKNNFunctions(processed));

        // Test processing logic with has_vector = false
        params.put("has_vector", false);
        processed = KNNMustacheScript.processTemplate(userTemplate, params);
        assertEquals("0", processed);
        assertFalse("Processed result should not contain k-NN function", KNNMustacheScript.containsKNNFunctions(processed));

        // Test processing logic with missing has_vector (should use inverted section)
        params.remove("has_vector");
        processed = KNNMustacheScript.processTemplate(userTemplate, params);
        assertEquals("0", processed);
    }

    public void testMustacheNestedConditionsAdvanced() {
        Map<String, Object> params = new HashMap<>();
        params.put("has_vector", true);
        params.put("use_cosine", true);

        String nestedTemplate =
            "{{#has_vector}}{{#use_cosine}}cosineSimilarity{{/use_cosine}}{{^use_cosine}}l2Distance{{/use_cosine}}(params.query, doc['field']){{/has_vector}}{{^has_vector}}0{{/has_vector}}";
        String processed = KNNMustacheScript.processTemplate(nestedTemplate, params);
        assertEquals("cosineSimilarity(params.query, doc['field'])", processed);

        params.put("use_cosine", false);
        processed = KNNMustacheScript.processTemplate(nestedTemplate, params);
        assertEquals("l2Distance(params.query, doc['field'])", processed);

        // Test dot notation
        Map<String, Object> nestedParams = new HashMap<>();
        nestedParams.put("query_vector", Arrays.asList(0.1, 0.2, 0.3));
        params.put("params", nestedParams);
        String result = KNNMustacheScript.processTemplate("Vector: {{params.query_vector}}", params);
        assertEquals("Vector: [0.1, 0.2, 0.3]", result);
    }

    public void testMustacheWithAllKNNFunctions() {
        String[] functions = { "cosineSimilarity", "l2Distance", "l2Squared", "innerProduct", "hamming", "l1Norm", "lInfNorm" };
        KNNScoringScriptEngine engine = new KNNScoringScriptEngine();

        // Test all functions with Mustache templates and factory validation
        for (String function : functions) {
            String template = "{{#enabled}}" + function + "(params.query, doc['field']){{/enabled}}{{^enabled}}0{{/enabled}}";
            ScoreScript.Factory factory = engine.compile("function_test", template, ScoreScript.CONTEXT, new HashMap<>());
            assertNotNull("Template with " + function + " should compile", factory);
            assertTrue("Should be Mustache factory", factory instanceof KNNScoreScriptFactory);
        }

        // Test factory validation (should reject scripts without k-NN functions or Mustache syntax)
        expectThrows(IllegalArgumentException.class, () -> new KNNScoreScriptFactory("knn_score")); // old syntax not allowed in new factory
        expectThrows(IllegalArgumentException.class, () -> new KNNScoreScriptFactory("plain text")); // no k-NN or Mustache
        expectThrows(IllegalArgumentException.class, () -> new KNNScoreScriptFactory("Math.max(1, 2)")); // no k-NN or Mustache

        // Valid Mustache templates and k-NN functions should work
        try {
            new KNNScoreScriptFactory("{{#condition}}text{{/condition}}"); // Valid Mustache
            new KNNScoreScriptFactory("{{variable}}"); // Valid Mustache
            new KNNScoreScriptFactory("cosineSimilarity(params.query, doc['field'])"); // Valid k-NN function
        } catch (Exception e) {
            fail("Valid scripts should not throw exceptions: " + e.getMessage());
        }
    }

    public void testMustacheEdgeCasesAndSpecialCharacters() {
        Map<String, Object> params = new HashMap<>();

        // Test edge cases
        assertEquals("", KNNMustacheScript.processTemplate("", params));
        assertEquals("plain text", KNNMustacheScript.processTemplate("plain text", params));
        assertEquals("{{#unclosed", KNNMustacheScript.processTemplate("{{#unclosed", params));

        // Test with empty params
        String result = KNNMustacheScript.processTemplate("{{missing}}", new HashMap<>());
        assertEquals("", result);

        // Test special characters
        params.put("field_name", "my-field_with.special:chars");
        result = KNNMustacheScript.processTemplate("doc['{{field_name}}']", params);
        assertEquals("doc['my-field_with.special:chars']", result);
    }

    // ========== Backward Compatibility and Mixed Usage ==========

    public void testBackwardCompatibilityAllScriptTypes() {
        KNNScoringScriptEngine engine = new KNNScoringScriptEngine();

        // Original syntax should still work
        ScoreScript.Factory oldFactory = engine.compile("old", "knn_score", ScoreScript.CONTEXT, new HashMap<>());
        assertTrue(oldFactory instanceof KNNScoreScriptFactory);

        // Function syntax should work
        ScoreScript.Factory functionFactory = engine.compile(
            "function",
            "cosineSimilarity(params.query, doc['field'])",
            ScoreScript.CONTEXT,
            new HashMap<>()
        );
        assertTrue(functionFactory instanceof KNNScoreScriptFactory);

        // Mustache syntax should work
        ScoreScript.Factory mustacheFactory = engine.compile(
            "mustache",
            "{{#condition}}cosineSimilarity(params.query, doc['field']){{/condition}}",
            ScoreScript.CONTEXT,
            new HashMap<>()
        );
        assertTrue(mustacheFactory instanceof KNNScoreScriptFactory);

        // Pure Mustache without k-NN functions should also work
        String pureMustache = "{{#condition}}1{{/condition}}{{^condition}}0{{/condition}}";
        ScoreScript.Factory pureMustacheFactory = engine.compile("pure_mustache", pureMustache, ScoreScript.CONTEXT, new HashMap<>());
        assertNotNull("Pure Mustache should compile", pureMustacheFactory);
        assertTrue("Should be Mustache factory", pureMustacheFactory instanceof KNNScoreScriptFactory);

        // All should use the same enhanced KNNScoreScriptFactory (unified approach)
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, oldFactory.getClass());
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, functionFactory.getClass());
        assertEquals("All should use KNNScoreScriptFactory", KNNScoreScriptFactory.class, mustacheFactory.getClass());
    }
}
