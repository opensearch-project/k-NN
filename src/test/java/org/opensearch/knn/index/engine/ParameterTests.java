/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.engine.Parameter.IntegerParameter;
import org.opensearch.knn.index.engine.Parameter.StringParameter;
import org.opensearch.knn.index.engine.Parameter.MethodComponentContextParameter;

import java.util.Map;

public class ParameterTests extends KNNTestCase {
    /**
     * Test default default value getter
     */
    public void testGetDefaultValue() {
        String defaultValue = "test-default";
        Parameter<String> parameter = new Parameter<String>("test", defaultValue, (v, context) -> true) {
            @Override
            public ValidationException validate(Object value, KNNMethodConfigContext context) {
                return null;
            }
        };

        assertEquals(defaultValue, parameter.getDefaultValue());
    }

    /**
     * Test integer parameter validate
     */
    public void testIntegerParameter_validate() {
        final IntegerParameter parameter = new IntegerParameter("test", 1, (v, context) -> v > 0);

        // Invalid type
        assertNotNull(parameter.validate("String", KNNMethodConfigContext.EMPTY));

        // Invalid value
        assertNotNull(parameter.validate(-1, KNNMethodConfigContext.EMPTY));

        // valid value
        assertNull(parameter.validate(12, KNNMethodConfigContext.EMPTY));
    }

    /**
     * Test integer parameter validate
     */
    public void testIntegerParameter_validateWithContext() {
        final IntegerParameter parameter = new IntegerParameter("test", 1, (v, context) -> v > 0 && v > context.getDimension().orElse(0));

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder().dimension(0).build();

        // Invalid type
        assertNotNull(parameter.validate("String", knnMethodConfigContext));

        // Invalid value
        assertNotNull(parameter.validate(-1, knnMethodConfigContext));

        // valid value
        assertNull(parameter.validate(12, knnMethodConfigContext));
    }

    public void testStringParameter_validate() {
        final StringParameter parameter = new StringParameter("test_parameter", "default_value", (v, context) -> "test".equals(v));

        // Invalid type
        assertNotNull(parameter.validate(5, KNNMethodConfigContext.EMPTY));

        // null
        assertNotNull(parameter.validate(null, KNNMethodConfigContext.EMPTY));

        // valid value
        assertNull(parameter.validate("test", KNNMethodConfigContext.EMPTY));
    }

    public void testStringParameter_validateWithData() {
        final StringParameter parameter = new StringParameter("test_parameter", "default_value", (v, context) -> {
            if (context.getDimension().orElse(1) > 0) {
                return "test".equals(v);
            }
            return false;
        });

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder().dimension(1).build();

        // Invalid type
        assertNotNull(parameter.validate(5, knnMethodConfigContext));

        // null
        assertNotNull(parameter.validate(null, knnMethodConfigContext));

        // valid value
        assertNull(parameter.validate("test", knnMethodConfigContext));

        knnMethodConfigContext.setDimension(0);

        // invalid value
        assertNotNull(parameter.validate("test", knnMethodConfigContext));
    }

    public void testDoubleParameter_validate() {
        final Parameter.DoubleParameter parameter = new Parameter.DoubleParameter("test_parameter", 1.0, (v, context) -> v >= 0);

        // valid value
        assertNull(parameter.validate(0.9, KNNMethodConfigContext.EMPTY));

        // Invalid type
        assertNotNull(parameter.validate(true, KNNMethodConfigContext.EMPTY));

        // Invalid type
        assertNotNull(parameter.validate(-1, KNNMethodConfigContext.EMPTY));

    }

    public void testDoubleParameter_validateWithData() {
        final Parameter.DoubleParameter parameter = new Parameter.DoubleParameter(
            "test",
            1.0,
            (v, context) -> v > 0 && v > context.getDimension().orElse(0)
        );

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder().dimension(0).build();

        // Invalid type
        assertNotNull(parameter.validate("String", knnMethodConfigContext));

        // Invalid value
        assertNotNull(parameter.validate(-1, knnMethodConfigContext));

        // valid value
        assertNull(parameter.validate(1.2, knnMethodConfigContext));
    }

    public void testMethodComponentContextParameter_validate() {
        String methodComponentName1 = "method-1";
        String parameterKey1 = "parameter_key_1";
        Integer parameterValue1 = 12;

        Map<String, Object> defaultParameterMap = ImmutableMap.of(parameterKey1, parameterValue1);
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodComponentName1, defaultParameterMap);

        Map<String, MethodComponent> methodComponentMap = ImmutableMap.of(
            methodComponentName1,
            MethodComponent.Builder.builder(parameterKey1)
                .addParameter(parameterKey1, new IntegerParameter(parameterKey1, 1, (v, context) -> v > 0))
                .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(
            "test",
            methodComponentContext,
            methodComponentMap
        );

        // Invalid type
        assertNotNull(parameter.validate(17, KNNMethodConfigContext.EMPTY));
        assertNotNull(parameter.validate("invalid-value", KNNMethodConfigContext.EMPTY));

        // Invalid value
        String invalidMethodComponentName = "invalid-method";
        MethodComponentContext invalidMethodComponentContext1 = new MethodComponentContext(invalidMethodComponentName, defaultParameterMap);
        assertNotNull(parameter.validate(invalidMethodComponentContext1, KNNMethodConfigContext.EMPTY));

        String invalidParameterKey = "invalid-parameter";
        Map<String, Object> invalidParameterMap1 = ImmutableMap.of(invalidParameterKey, parameterValue1);
        MethodComponentContext invalidMethodComponentContext2 = new MethodComponentContext(methodComponentName1, invalidParameterMap1);
        assertNotNull(parameter.validate(invalidMethodComponentContext2, KNNMethodConfigContext.EMPTY));

        String invalidParameterValue = "invalid-value";
        Map<String, Object> invalidParameterMap2 = ImmutableMap.of(parameterKey1, invalidParameterValue);
        MethodComponentContext invalidMethodComponentContext3 = new MethodComponentContext(methodComponentName1, invalidParameterMap2);
        assertNotNull(parameter.validate(invalidMethodComponentContext3, KNNMethodConfigContext.EMPTY));

        // valid value
        assertNull(parameter.validate(methodComponentContext, KNNMethodConfigContext.EMPTY));
    }

    public void testMethodComponentContextParameter_validateWithData() {
        String methodComponentName1 = "method-1";
        String parameterKey1 = "parameter_key_1";
        Integer parameterValue1 = 12;

        Map<String, Object> defaultParameterMap = ImmutableMap.of(parameterKey1, parameterValue1);
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodComponentName1, defaultParameterMap);

        Map<String, MethodComponent> methodComponentMap = ImmutableMap.of(
            methodComponentName1,
            MethodComponent.Builder.builder(parameterKey1)
                .addParameter(
                    parameterKey1,
                    new IntegerParameter(parameterKey1, 1, (v, context) -> v > 0 && v > context.getDimension().orElse(0))
                )
                .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(
            "test",
            methodComponentContext,
            methodComponentMap
        );

        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder().dimension(0).build();

        // Invalid type
        assertNotNull(parameter.validate("invalid-value", knnMethodConfigContext));

        // Invalid value
        String invalidMethodComponentName = "invalid-method";
        MethodComponentContext invalidMethodComponentContext1 = new MethodComponentContext(invalidMethodComponentName, defaultParameterMap);
        assertNotNull(parameter.validate(invalidMethodComponentContext1, knnMethodConfigContext));

        String invalidParameterKey = "invalid-parameter";
        Map<String, Object> invalidParameterMap1 = ImmutableMap.of(invalidParameterKey, parameterValue1);
        MethodComponentContext invalidMethodComponentContext2 = new MethodComponentContext(methodComponentName1, invalidParameterMap1);
        assertNotNull(parameter.validate(invalidMethodComponentContext2, knnMethodConfigContext));

        String invalidParameterValue = "invalid-value";
        Map<String, Object> invalidParameterMap2 = ImmutableMap.of(parameterKey1, invalidParameterValue);
        MethodComponentContext invalidMethodComponentContext3 = new MethodComponentContext(methodComponentName1, invalidParameterMap2);
        assertNotNull(parameter.validate(invalidMethodComponentContext3, knnMethodConfigContext));

        // valid value
        assertNull(parameter.validate(methodComponentContext, knnMethodConfigContext));
    }

    public void testMethodComponentContextParameter_getMethodComponent() {
        String methodComponentName1 = "method-1";
        String parameterKey1 = "parameter_key_1";
        Integer parameterValue1 = 12;

        Map<String, Object> defaultParameterMap = ImmutableMap.of(parameterKey1, parameterValue1);
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodComponentName1, defaultParameterMap);

        Map<String, MethodComponent> methodComponentMap = ImmutableMap.of(
            methodComponentName1,
            MethodComponent.Builder.builder(parameterKey1)
                .addParameter(parameterKey1, new IntegerParameter(parameterKey1, 1, (v, context) -> v > 0))
                .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(
            "test",
            methodComponentContext,
            methodComponentMap
        );

        // Test when method component is available
        assertEquals(methodComponentMap.get(methodComponentName1), parameter.getMethodComponent(methodComponentName1));

        // test when method component is not available
        String invalidMethod = "invalid-method";
        assertNull(parameter.getMethodComponent(invalidMethod));
    }
}
