/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.script;

import org.apache.lucene.index.LeafReaderContext;
import org.opensearch.knn.index.KNNVectorScriptDocValues;
import org.opensearch.script.ScoreScript;
import org.opensearch.search.lookup.SearchLookup;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Mustache script that handles both direct k-NN functions and Mustache templates containing k-NN functions.
 * This extends ScoreScript and consolidates the execution logic that was duplicated across separate script classes.
 * Also includes function parsing and Mustache processing utilities to eliminate separate utility classes.
 */
public class KNNMustacheScript extends ScoreScript {

    // ========== Function Parser Utilities ==========

    private static final Pattern FUNCTION_PATTERN = Pattern.compile(
        "(cosineSimilarity|l2Distance|l2Squared|innerProduct|hamming|l1Norm|lInfNorm)\\s*\\(\\s*([^,]+)\\s*,\\s*([^)]+)\\s*\\)"
    );

    /**
     * Checks if the script contains any k-NN function calls.
     */
    public static boolean containsKNNFunctions(String script) {
        return FUNCTION_PATTERN.matcher(script).find();
    }

    /**
     * Parse the first k-NN function call found in the script.
     */
    public static FunctionCall parseFunction(String script) {
        Matcher matcher = FUNCTION_PATTERN.matcher(script);
        if (matcher.find()) {
            return new FunctionCall(matcher.group(1), matcher.group(2).trim(), matcher.group(3).trim());
        }
        return null;
    }

    /**
     * Parsed function call details.
     */
    public static class FunctionCall {
        private final String functionName;
        private final String queryParam;
        private final String docField;

        public FunctionCall(String functionName, String queryParam, String docField) {
            this.functionName = functionName;
            this.queryParam = queryParam;
            this.docField = docField;
        }

        public String getFunctionName() {
            return functionName;
        }

        public String getQueryParam() {
            return queryParam;
        }

        public String getDocField() {
            return docField;
        }

        /**
         * Extract field name from doc['field'] or doc["field"] syntax.
         */
        public String extractFieldName() {
            String field = docField.trim();
            if (field.startsWith("doc[")) {
                // Extract field name from doc['field'] or doc["field"]
                int start = field.indexOf('[') + 1;
                int end = field.lastIndexOf(']');
                if (start > 0 && end > start) {
                    String fieldRef = field.substring(start, end).trim();
                    // Remove quotes if present
                    if ((fieldRef.startsWith("'") && fieldRef.endsWith("'")) || (fieldRef.startsWith("\"") && fieldRef.endsWith("\""))) {
                        return fieldRef.substring(1, fieldRef.length() - 1);
                    }
                    return fieldRef;
                }
            }
            return field;
        }

        /**
         * Extract parameter name from params.name syntax.
         */
        public String extractParamName() {
            String param = queryParam.trim();
            if (param.startsWith("params.")) {
                return param.substring(7);
            }
            return param;
        }
    }

    // ========== Mustache Processor Utilities ==========

    // Pattern for conditional sections: {{#condition}}content{{/condition}}
    private static final Pattern SECTION_PATTERN = Pattern.compile("\\{\\{#([^}]+)\\}\\}(.*?)\\{\\{/\\1\\}\\}", Pattern.DOTALL);

    // Pattern for inverted sections: {{^condition}}content{{/condition}}
    private static final Pattern INVERTED_SECTION_PATTERN = Pattern.compile("\\{\\{\\^([^}]+)\\}\\}(.*?)\\{\\{/\\1\\}\\}", Pattern.DOTALL);

    // Pattern for variable substitution: {{variable}}
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\{\\{([^#^/][^}]*)\\}\\}");

    /**
     * Checks if the template contains Mustache syntax.
     */
    public static boolean isMustacheTemplate(String template) {
        return template.contains("{{") && template.contains("}}");
    }

    /**
     * Process Mustache template with given parameters.
     */
    public static String processTemplate(String template, Map<String, Object> params) {
        if (!isMustacheTemplate(template)) {
            return template;
        }

        String result = template;

        // Process conditional sections {{#condition}}...{{/condition}}
        result = processSections(result, params, false);

        // Process inverted sections {{^condition}}...{{/condition}}
        result = processSections(result, params, true);

        // Process variable substitution {{variable}}
        result = processVariables(result, params);

        return result;
    }

    /**
     * Process conditional or inverted sections based on parameter values.
     */
    private static String processSections(String template, Map<String, Object> params, boolean inverted) {
        Pattern pattern = inverted ? INVERTED_SECTION_PATTERN : SECTION_PATTERN;
        Matcher matcher = pattern.matcher(template);
        StringBuilder result = new StringBuilder();
        int lastEnd = 0;

        while (matcher.find()) {
            String condition = matcher.group(1).trim();
            String content = matcher.group(2);

            // Append text before this section
            result.append(template.substring(lastEnd, matcher.start()));

            // Evaluate condition
            boolean conditionValue = evaluateCondition(condition, params);

            // For inverted sections, flip the condition
            if (inverted) {
                conditionValue = !conditionValue;
            }

            // Include content if condition is true, and recursively process it
            if (conditionValue) {
                // Recursively process nested sections and variables in the content
                String processedContent = processTemplate(content, params);
                result.append(processedContent);
            }

            lastEnd = matcher.end();
        }

        // Append remaining text
        result.append(template.substring(lastEnd));
        return result.toString();
    }

    /**
     * Process variable substitution in the template.
     */
    private static String processVariables(String template, Map<String, Object> params) {
        Matcher matcher = VARIABLE_PATTERN.matcher(template);
        StringBuilder result = new StringBuilder();
        int lastEnd = 0;

        while (matcher.find()) {
            String variableName = matcher.group(1).trim();

            // Append text before this variable
            result.append(template.substring(lastEnd, matcher.start()));

            // Substitute variable value
            Object value = getVariableValue(variableName, params);
            result.append(value != null ? value.toString() : "");

            lastEnd = matcher.end();
        }

        // Append remaining text
        result.append(template.substring(lastEnd));
        return result.toString();
    }

    /**
     * Evaluate condition based on parameter values.
     */
    private static boolean evaluateCondition(String condition, Map<String, Object> params) {
        Object value = getVariableValue(condition, params);

        if (value == null) {
            return false;
        }

        if (value instanceof Boolean) {
            return (Boolean) value;
        }

        if (value instanceof Number) {
            return ((Number) value).doubleValue() != 0.0;
        }

        if (value instanceof String) {
            return !((String) value).isEmpty();
        }

        if (value instanceof java.util.Collection) {
            return !((java.util.Collection<?>) value).isEmpty();
        }

        if (value.getClass().isArray()) {
            return java.lang.reflect.Array.getLength(value) > 0;
        }

        // For any other object type, existence means true
        return true;
    }

    /**
     * Get variable value from parameters, supporting dot notation.
     */
    private static Object getVariableValue(String variableName, Map<String, Object> params) {
        if (variableName.contains(".")) {
            // Handle dot notation like "params.query_vector"
            String[] parts = variableName.split("\\.", 2);
            String rootKey = parts[0];
            String remainingPath = parts[1];

            Object rootValue = params.get(rootKey);
            if (rootValue instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> nestedMap = (Map<String, Object>) rootValue;
                return getVariableValue(remainingPath, nestedMap);
            }
            return null;
        } else {
            return params.get(variableName);
        }
    }

    // ========== Script Execution Logic ==========

    private final String scriptSource;
    private final Map<String, Object> scriptParams;
    private final boolean isMustacheTemplate;

    public KNNMustacheScript(String scriptSource, Map<String, Object> params, SearchLookup lookup, LeafReaderContext leafContext)
        throws IOException {
        super(params, lookup, null, leafContext);
        this.scriptSource = scriptSource;
        this.scriptParams = params;
        this.isMustacheTemplate = isMustacheTemplate(scriptSource);
    }

    @Override
    @SuppressWarnings("unchecked")
    public double execute(ScoreScript.ExplanationHolder explanationHolder) {
        try {
            // Step 1: Process the script source (Mustache templates are processed, others pass through)
            String processedScript = isMustacheTemplate ? processTemplate(scriptSource, scriptParams) : scriptSource;

            // Step 2: Handle the processed result
            if (containsKNNFunctions(processedScript)) {
                return executeKNNFunction(processedScript);
            } else {
                // For Mustache templates that evaluate to numeric values
                return parseNumericResult(processedScript);
            }

        } catch (Exception e) {
            throw new RuntimeException("Error executing k-NN script: " + scriptSource, e);
        }
    }

    /**
     * Executes a k-NN function call using the unified execution logic.
     */
    private double executeKNNFunction(String functionScript) {
        FunctionCall functionCall = parseFunction(functionScript);
        if (functionCall == null) {
            throw new IllegalArgumentException("Invalid k-NN function in script: " + functionScript);
        }

        // Get the query vector from params
        String paramName = functionCall.extractParamName();
        Object queryParam = scriptParams.get(paramName);

        if (!(queryParam instanceof List)) {
            throw new IllegalArgumentException("Query parameter must be a list: " + paramName);
        }

        List<Number> queryVector = (List<Number>) queryParam;

        // Get the document field
        String fieldName = functionCall.extractFieldName();
        KNNVectorScriptDocValues<?> docValues = (KNNVectorScriptDocValues<?>) getDoc().get(fieldName);

        if (docValues.isEmpty()) {
            return 0.0;
        }

        // Execute the appropriate scoring function
        String functionName = functionCall.getFunctionName();
        switch (functionName) {
            case "cosineSimilarity":
                return KNNScoringUtil.cosineSimilarity(queryVector, docValues);
            case "l2Distance":
            case "l2Squared":
                return KNNScoringUtil.l2Squared(queryVector, docValues);
            case "innerProduct":
                return KNNScoringUtil.innerProduct(queryVector, docValues);
            case "hamming":
                return KNNScoringUtil.hamming(queryVector, docValues);
            case "l1Norm":
                return KNNScoringUtil.l1Norm(queryVector, docValues);
            case "lInfNorm":
                return KNNScoringUtil.lInfNorm(queryVector, docValues);
            default:
                throw new IllegalArgumentException("Unsupported k-NN function: " + functionName);
        }
    }

    /**
     * Parses numeric results from Mustache template processing.
     */
    private double parseNumericResult(String result) {
        try {
            return Double.parseDouble(result.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Script must result in either a k-NN function call or a numeric value. Got: " + result);
        }
    }
}
