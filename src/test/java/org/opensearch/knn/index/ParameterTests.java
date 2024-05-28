/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.Parameter.IntegerParameter;
import org.opensearch.knn.index.Parameter.StringParameter;
import org.opensearch.knn.index.Parameter.MethodComponentContextParameter;
import org.opensearch.knn.training.TrainingDataSpec;

import java.util.Map;

public class ParameterTests extends KNNTestCase {
    /**
     * Test default default value getter
     */
    public void testGetDefaultValue() {
        String defaultValue = "test-default";
        Parameter<String> parameter = new Parameter<String>("test", defaultValue, v -> true) {
            @Override
            public ValidationException validate(Object value) {
                return null;
            }

            @Override
            public ValidationException validateWithData(Object value, TrainingDataSpec trainingDataSpec) {
                return null;
            }

        };

        assertEquals(defaultValue, parameter.getDefaultValue());
    }

    /**
     * Test integer parameter validate
     */
    public void testIntegerParameter_validate() {
        final IntegerParameter parameter = new IntegerParameter("test", 1, v -> v > 0);

        // Invalid type
        assertNotNull(parameter.validate("String"));

        // Invalid value
        assertNotNull(parameter.validate(-1));

        // valid value
        assertNull(parameter.validate(12));
    }

    /**
     * Test integer parameter validate
     */
    public void testIntegerParameter_validateWithData() {
        final IntegerParameter parameter = new IntegerParameter(
            "test",
            1,
            v -> v > 0,
            (v, trainingDataSpec) -> v > trainingDataSpec.getDimension()
        );

        TrainingDataSpec testTrainingDataSpec = new TrainingDataSpec(0);

        // Invalid type
        assertNotNull(parameter.validateWithData("String", testTrainingDataSpec));

        // Invalid value
        assertNotNull(parameter.validateWithData(-1, testTrainingDataSpec));

        // valid value
        assertNull(parameter.validateWithData(12, testTrainingDataSpec));
    }

    public void testStringParameter_validate() {
        final StringParameter parameter = new StringParameter("test_parameter", "default_value", v -> "test".equals(v));

        // Invalid type
        assertNotNull(parameter.validate(5));

        // null
        assertNotNull(parameter.validate(null));

        // valid value
        assertNull(parameter.validate("test"));
    }

    public void testStringParameter_validateWithData() {
        final StringParameter parameter = new StringParameter(
            "test_parameter",
            "default_value",
            v -> "test".equals(v),
            (v, trainingDataSpec) -> {
                if (trainingDataSpec.getDimension() > 0) {
                    return "test".equals(v);
                }
                return false;
            }
        );

        TrainingDataSpec testTrainingDataSpec = new TrainingDataSpec(1);

        // Invalid type
        assertNotNull(parameter.validateWithData(5, testTrainingDataSpec));

        // null
        assertNotNull(parameter.validateWithData(null, testTrainingDataSpec));

        // valid value
        assertNull(parameter.validateWithData("test", testTrainingDataSpec));

        testTrainingDataSpec.setDimension(0);

        // invalid value
        assertNotNull(parameter.validateWithData("test", testTrainingDataSpec));
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
                .addParameter(parameterKey1, new IntegerParameter(parameterKey1, 1, v -> v > 0))
                .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(
            "test",
            methodComponentContext,
            methodComponentMap
        );

        // Invalid type
        assertNotNull(parameter.validate(17));
        assertNotNull(parameter.validate("invalid-value"));

        // Invalid value
        String invalidMethodComponentName = "invalid-method";
        MethodComponentContext invalidMethodComponentContext1 = new MethodComponentContext(invalidMethodComponentName, defaultParameterMap);
        assertNotNull(parameter.validate(invalidMethodComponentContext1));

        String invalidParameterKey = "invalid-parameter";
        Map<String, Object> invalidParameterMap1 = ImmutableMap.of(invalidParameterKey, parameterValue1);
        MethodComponentContext invalidMethodComponentContext2 = new MethodComponentContext(methodComponentName1, invalidParameterMap1);
        assertNotNull(parameter.validate(invalidMethodComponentContext2));

        String invalidParameterValue = "invalid-value";
        Map<String, Object> invalidParameterMap2 = ImmutableMap.of(parameterKey1, invalidParameterValue);
        MethodComponentContext invalidMethodComponentContext3 = new MethodComponentContext(methodComponentName1, invalidParameterMap2);
        assertNotNull(parameter.validate(invalidMethodComponentContext3));

        // valid value
        assertNull(parameter.validate(methodComponentContext));
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
                    new IntegerParameter(parameterKey1, 1, v -> v > 0, (v, trainingDataSpec) -> v > trainingDataSpec.getDimension())
                )
                .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(
            "test",
            methodComponentContext,
            methodComponentMap
        );

        TrainingDataSpec testTrainingDataSpec = new TrainingDataSpec(0);

        // Invalid type
        assertNotNull(parameter.validateWithData(17, testTrainingDataSpec));
        assertNotNull(parameter.validateWithData("invalid-value", testTrainingDataSpec));

        // Invalid value
        String invalidMethodComponentName = "invalid-method";
        MethodComponentContext invalidMethodComponentContext1 = new MethodComponentContext(invalidMethodComponentName, defaultParameterMap);
        assertNotNull(parameter.validateWithData(invalidMethodComponentContext1, testTrainingDataSpec));

        String invalidParameterKey = "invalid-parameter";
        Map<String, Object> invalidParameterMap1 = ImmutableMap.of(invalidParameterKey, parameterValue1);
        MethodComponentContext invalidMethodComponentContext2 = new MethodComponentContext(methodComponentName1, invalidParameterMap1);
        assertNotNull(parameter.validateWithData(invalidMethodComponentContext2, testTrainingDataSpec));

        String invalidParameterValue = "invalid-value";
        Map<String, Object> invalidParameterMap2 = ImmutableMap.of(parameterKey1, invalidParameterValue);
        MethodComponentContext invalidMethodComponentContext3 = new MethodComponentContext(methodComponentName1, invalidParameterMap2);
        assertNotNull(parameter.validateWithData(invalidMethodComponentContext3, testTrainingDataSpec));

        // valid value
        assertNull(parameter.validateWithData(methodComponentContext, testTrainingDataSpec));
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
                .addParameter(parameterKey1, new IntegerParameter(parameterKey1, 1, v -> v > 0))
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
