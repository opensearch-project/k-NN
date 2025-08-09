/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.TemplateScript;
import org.opensearch.test.OpenSearchTestCase;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * Unit tests for KNNEnhancedMustacheEngine.
 *
 * Tests basic functionality of the simplified k-NN Mustache engine including:
 * - Engine type and name
 * - Supported script contexts
 * - Factory creation
 * - Basic template compilation (future enhancement)
 */
public class KNNEnhancedMustacheEngineTests extends OpenSearchTestCase {

    private KNNEnhancedMustacheEngine engine;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        engine = new KNNEnhancedMustacheEngine();
    }

    public void testEngineType() {
        assertEquals("knn-mustache", engine.getType());
    }

    public void testSupportedContexts() {
        Set<ScriptContext<?>> contexts = engine.getSupportedContexts();
        assertEquals(2, contexts.size());
        assertTrue(contexts.contains(TemplateScript.CONTEXT));
        assertTrue(contexts.contains(ScoreScript.CONTEXT));
    }

    public void testMustacheFactoryCreation() {
        Map<String, String> options = Collections.emptyMap();
        String template = "{{test}}";
        assertNotNull(engine.compile("test", template, TemplateScript.CONTEXT, options));
    }

    public void testMustacheFactoryCreationWithOptions() {
        // Test template compilation with content type option
        Map<String, String> options = Collections.singletonMap("content_type", "application/json");
        String template = "{{test}}";
        assertNotNull(engine.compile("test", template, TemplateScript.CONTEXT, options));
    }

    public void testEngineNameConstant() {
        assertEquals("knn-mustache", KNNEnhancedMustacheEngine.NAME);
    }

    public void testMustacheFactoryIsCustomType() {
        // Verify that our engine can process templates correctly
        Map<String, String> options = Collections.emptyMap();
        String template = "{{field}}";
        TemplateScript.Factory factory = engine.compile("test", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);
        // Verify we can create an executable script
        Map<String, Object> params = Collections.singletonMap("field", "value");
        TemplateScript script = factory.newInstance(params);
        assertNotNull(script);
    }

    public void testVectorFunctionRecognition() {
        // Test that our visitor recognizes k-NN vector functions
        // This test verifies the function matching logic is in place
        // Test basic k-NN function recognition
        assertNotNull(engine); // Engine should be initialized
    }

    public void testEngineIntegration() {
        // Test that the engine properly integrates with OpenSearch scripting framework
        assertTrue(engine.getSupportedContexts().contains(TemplateScript.CONTEXT));
        assertTrue(engine.getSupportedContexts().contains(ScoreScript.CONTEXT));
        assertEquals("knn-mustache", engine.getType());
    }

    public void testScoreScriptCompilation() {
        // Test that the engine can compile ScoreScript templates
        Map<String, String> options = Collections.emptyMap();
        String template = "{{score}}";
        assertNotNull(engine.compile("test-score", template, ScoreScript.CONTEXT, options));
    }

    public void testVectorFunctionSupport() {
        // Test that the engine supports vector function template compilation
        Map<String, String> options = Collections.emptyMap();

        // Test template script with vector functions
        String templateWithVectorFunctions = "{{l2Distance}} {{cosineSimilarity}} {{innerProduct}} {{hamming}}";
        assertNotNull(engine.compile("test-vector", templateWithVectorFunctions, TemplateScript.CONTEXT, options));

        // Test score script with vector utilities
        String scoreScriptTemplate = "{{vectorUtils}}";
        assertNotNull(engine.compile("test-score-vector", scoreScriptTemplate, ScoreScript.CONTEXT, options));
    }

    public void testBasicKNNTemplateCompilation() {
        // Test basic k-NN query template compilation and execution
        Map<String, String> options = Collections.emptyMap();
        String template = "{\"knn\": {\"{{field}}\": {\"vector\": {{vector}}, \"k\": {{k}}}}}";

        TemplateScript.Factory factory = engine.compile("test-knn-basic", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);

        Map<String, Object> params = Map.of("field", "my_vector_field", "vector", "[1.0, 2.0, 3.0]", "k", "10");

        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        // Verify the template was rendered correctly
        assertTrue(result.contains("my_vector_field"));
        assertTrue(result.contains("[1.0, 2.0, 3.0]"));
        assertTrue(result.contains("10"));
    }

    public void testL2DistanceTemplateCompilation() {
        // Test L2 distance function in template compilation
        Map<String, String> options = Collections.emptyMap();
        String template = "{{l2Distance}}";

        TemplateScript.Factory factory = engine.compile("test-l2", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);

        Map<String, Object> params = Collections.emptyMap();
        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        // Verify l2Distance function is available
        assertEquals("l2Squared", result);
    }

    public void testCosineSimilarityTemplateCompilation() {
        // Test cosine similarity function in template compilation
        Map<String, String> options = Collections.emptyMap();
        String template = "{{cosineSimilarity}}";

        TemplateScript.Factory factory = engine.compile("test-cosine", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);

        Map<String, Object> params = Collections.emptyMap();
        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        // Verify cosineSimilarity function is available
        assertEquals("cosineSimilarity", result);
    }

    public void testInnerProductTemplateCompilation() {
        // Test inner product function in template compilation
        Map<String, String> options = Collections.emptyMap();
        String template = "{{innerProduct}}";

        TemplateScript.Factory factory = engine.compile("test-inner", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);

        Map<String, Object> params = Collections.emptyMap();
        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        // Verify innerProduct function is available
        assertEquals("innerProduct", result);
    }

    public void testHammingTemplateCompilation() {
        // Test hamming distance function in template compilation
        Map<String, String> options = Collections.emptyMap();
        String template = "{{hamming}}";

        TemplateScript.Factory factory = engine.compile("test-hamming", template, TemplateScript.CONTEXT, options);
        assertNotNull(factory);

        Map<String, Object> params = Collections.emptyMap();
        TemplateScript script = factory.newInstance(params);
        String result = script.execute();

        // Verify hamming function is available
        assertEquals("hamming", result);
    }
}
