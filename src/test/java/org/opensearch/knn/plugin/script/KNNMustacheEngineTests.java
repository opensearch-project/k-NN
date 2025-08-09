/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptException;
import org.opensearch.script.TemplateScript;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Unit tests for KNNMustacheEngine.
 *
 * Tests Mustache templating functionality for dynamic k-NN query generation including:
 * - Engine type and name validation
 * - Template script context support
 * - Factory creation and template compilation
 * - Dynamic k-NN query template execution
 * - Error handling and edge cases
 */
public class KNNMustacheEngineTests extends OpenSearchTestCase {

    private static final String KNN_TEMPLATE = "{\"knn\": {\"{{field}}\": {\"vector\": {{vector}}, \"k\": {{k}}}}}";
    private static final String COMPLEX_KNN_TEMPLATE =
        "{\"knn\": {\"{{field}}\": {\"vector\": {{vector}}, \"k\": {{k}}, \"max_distance\": {{max_distance}}, \"min_score\": {{min_score}}}}}";

    private KNNMustacheEngine engine;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        engine = new KNNMustacheEngine();
    }

    // Helper method to reduce boilerplate
    private void assertTemplateExecution(String template, Map<String, Object> params, String... expectedContents) {
        TemplateScript.Factory factory = engine.compile("test", template, TemplateScript.CONTEXT, Collections.emptyMap());
        assertNotNull(factory);
        String result = factory.newInstance(params).execute();
        for (String expected : expectedContents) {
            assertTrue("Result should contain: " + expected, result.contains(expected));
        }
    }

    public void testEngineBasics() {
        assertEquals("knn-mustache", engine.getType());
        assertEquals("knn-mustache", KNNMustacheEngine.NAME);

        Set<ScriptContext<?>> contexts = engine.getSupportedContexts();
        assertEquals(1, contexts.size());
        assertTrue(contexts.contains(TemplateScript.CONTEXT));
    }

    public void testTemplateCompilation() {
        Map<String, String> options = Collections.emptyMap();
        String template = "{{test}}";

        // Basic compilation
        assertNotNull(engine.compile("test", template, TemplateScript.CONTEXT, options));

        // With options
        Map<String, String> optionsWithContentType = Collections.singletonMap("content_type", "application/json");
        assertNotNull(engine.compile("test", template, TemplateScript.CONTEXT, optionsWithContentType));

        // Null template name handling
        assertNotNull(engine.compile(null, template, TemplateScript.CONTEXT, options));
    }

    public void testBasicKNNTemplates() {
        // Basic k-NN template
        assertTemplateExecution(
            KNN_TEMPLATE,
            Map.of("field", "my_vector_field", "vector", "[1.0, 2.0, 3.0]", "k", "10"),
            "my_vector_field",
            "[1.0, 2.0, 3.0]",
            "10"
        );

        // Complex k-NN template with multiple parameters
        assertTemplateExecution(
            COMPLEX_KNN_TEMPLATE,
            Map.of("field", "embeddings", "vector", "[0.1, 0.2, 0.3]", "k", "5", "max_distance", "1.0", "min_score", "0.8"),
            "embeddings",
            "[0.1, 0.2, 0.3]",
            "5",
            "1.0",
            "0.8"
        );
    }

    public void testAdvancedKNNTemplates() {
        // Multi-field template
        String multiFieldTemplate =
            "{\"bool\": {\"should\": [{\"knn\": {\"{{field1}}\": {\"vector\": {{vector1}}, \"k\": {{k}}}}}, {\"knn\": {\"{{field2}}\": {\"vector\": {{vector2}}, \"k\": {{k}}}}}]}}";
        assertTemplateExecution(
            multiFieldTemplate,
            Map.of(
                "field1",
                "title_embeddings",
                "field2",
                "content_embeddings",
                "vector1",
                "[0.1, 0.2]",
                "vector2",
                "[0.3, 0.4]",
                "k",
                "10"
            ),
            "title_embeddings",
            "content_embeddings",
            "[0.1, 0.2]",
            "[0.3, 0.4]",
            "10"
        );

        // Filtered k-NN template
        String filteredTemplate =
            "{\"knn\": {\"{{field}}\": {\"vector\": {{vector}}, \"k\": {{k}}, \"filter\": {\"term\": {\"{{filter_field}}\": \"{{filter_value}}\"}}}}}";
        assertTemplateExecution(
            filteredTemplate,
            Map.of(
                "field",
                "vector_field",
                "vector",
                "[1.0, 2.0, 3.0, 4.0]",
                "k",
                "20",
                "filter_field",
                "category",
                "filter_value",
                "electronics"
            ),
            "vector_field",
            "[1.0, 2.0, 3.0, 4.0]",
            "20",
            "category",
            "electronics"
        );
    }

    public void testErrorHandling() {
        Map<String, String> invalidTemplates = Map.of(
            "unclosed-tag",
            "{{unclosed_tag",
            "mismatched-tags",
            "{{#section}}{{field}}{{/wrong_section}}"
        );

        invalidTemplates.forEach(
            (name, template) -> expectThrows(
                ScriptException.class,
                () -> engine.compile(name, template, TemplateScript.CONTEXT, Collections.emptyMap())
            )
        );

        // Unsupported script context
        expectThrows(IllegalArgumentException.class, () -> {
            ScriptContext<?> unsupportedContext = new ScriptContext<>("unsupported", Object.class);
            engine.compile("test", "{{test}}", unsupportedContext, Collections.emptyMap());
        });
    }

    public void testParameterHandling() {
        TemplateScript.Factory factory = engine.compile("test-params", KNN_TEMPLATE, TemplateScript.CONTEXT, Collections.emptyMap());

        // Missing parameter (renders as empty string)
        Map<String, Object> incompleteParams = Map.of("vector", "[1.0, 2.0]", "k", "5");
        String result = factory.newInstance(incompleteParams).execute();
        assertTrue(result.contains("[1.0, 2.0]"));
        assertTrue(result.contains("5"));

        // Null parameter handling
        Map<String, Object> paramsWithNull = new HashMap<>();
        paramsWithNull.put("field", "vector_field");
        paramsWithNull.put("vector", "[1.0, 2.0]");
        paramsWithNull.put("k", null);

        result = factory.newInstance(paramsWithNull).execute();
        assertTrue(result.contains("vector_field"));
        assertTrue(result.contains("[1.0, 2.0]"));
    }

    public void testSpecialCases() {
        TemplateScript.Factory emptyFactory = engine.compile("empty", "", TemplateScript.CONTEXT, Collections.emptyMap());
        assertEquals("", emptyFactory.newInstance(Collections.emptyMap()).execute());

        String specialTemplate = "{\"knn\": {\"{{field}}\": {\"query\": \"{{query_text}}\", \"k\": {{k}}}}}";
        String specialText = "special chars: éñ中文 & quotes";  // Remove problematic quotes

        TemplateScript.Factory factory = engine.compile("special-chars", specialTemplate, TemplateScript.CONTEXT, Collections.emptyMap());
        assertNotNull(factory);

        Map<String, Object> params = Map.of("field", "content_field", "query_text", specialText, "k", "10");
        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        assertTrue("Result should contain field name", result.contains("content_field"));
        assertTrue("Result should contain k value", result.contains("10"));
        assertTrue("Result should contain some form of the special text", result.contains("éñ中文") || result.contains("special chars"));
    }

    public void testLargeTemplateAndConcurrency() throws InterruptedException {
        // Large template stress test
        StringBuilder templateBuilder = new StringBuilder("{\"knn\": {\"{{field}}\": {\"vector\": {{vector}}, \"k\": {{k}}");
        Map<String, Object> manyParams = new HashMap<>();
        manyParams.put("field", "test_field");
        manyParams.put("vector", "[1.0, 2.0, 3.0]");
        manyParams.put("k", "20");

        for (int i = 0; i < 50; i++) {
            templateBuilder.append(", \"filter").append(i).append("\": \"{{filter").append(i).append("}}\"");
            manyParams.put("filter" + i, "value" + i);
        }
        templateBuilder.append("}}}");
        assertTemplateExecution(templateBuilder.toString(), manyParams, "test_field", "[1.0, 2.0, 3.0]", "value0", "value49");

        // Concurrent execution test
        TemplateScript.Factory factory = engine.compile("concurrent-test", KNN_TEMPLATE, TemplateScript.CONTEXT, Collections.emptyMap());
        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];
        boolean[] results = new boolean[numThreads];

        for (int i = 0; i < numThreads; i++) {
            final int threadIndex = i;
            threads[i] = new Thread(() -> {
                try {
                    Map<String, Object> params = Map.of(
                        "field",
                        "field_" + threadIndex,
                        "vector",
                        "[" + threadIndex + ".0, " + (threadIndex + 1) + ".0]",
                        "k",
                        String.valueOf(threadIndex + 5)
                    );
                    String result = factory.newInstance(params).execute();
                    results[threadIndex] = result.contains("field_" + threadIndex)
                        && result.contains(threadIndex + ".0")
                        && result.contains(String.valueOf(threadIndex + 5));
                } catch (Exception e) {
                    results[threadIndex] = false;
                }
            });
        }

        for (Thread thread : threads)
            thread.start();
        for (Thread thread : threads)
            thread.join();
        for (boolean result : results) {
            assertTrue("Concurrent template execution failed", result);
        }
    }

    public void testEngineIntegration() {
        // Verify factory creates executable scripts
        TemplateScript.Factory factory = engine.compile("test", "{{field}}", TemplateScript.CONTEXT, Collections.emptyMap());
        assertNotNull(factory);

        TemplateScript script = factory.newInstance(Collections.singletonMap("field", "value"));
        assertNotNull(script);
        assertEquals("value", script.execute());
    }
}
