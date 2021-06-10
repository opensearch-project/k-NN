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
import org.opensearch.knn.index.Parameter.MethodComponentContextParameter;

import java.util.Map;

public class ParameterTests extends KNNTestCase {
    /**
     * Test default default value getter
     */
    public void testGetDefaultValue() {
        String defaultValue = "test-default";
        Parameter<String> parameter = new Parameter<String>(defaultValue, v -> true) {
            @Override
            public void validate(Object value) {}
        };

        assertEquals(defaultValue, parameter.getDefaultValue());
    }

    /**
     * Test integer parameter validate
     */
    public void testIntegerParameter_validate() {
        final IntegerParameter parameter = new IntegerParameter(1,
                v -> v > 0);

        // Invalid type
        expectThrows(ValidationException.class, () -> parameter.validate("String"));

        // Invalid value
        expectThrows(ValidationException.class, () -> parameter.validate(-1));

        // valid value
        parameter.validate(12);
    }

    public void testMethodComponentContextParameter_validate() {
        String methodComponentName1 = "method-1";
        String parameterKey1 = "parameter_key_1";
        Integer parameterValue1 = 12;

        Map<String, Object> defaultParameterMap = ImmutableMap.of(parameterKey1, parameterValue1);
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodComponentName1,
                defaultParameterMap);

        Map<String, MethodComponent> methodComponentMap = ImmutableMap.of(
                methodComponentName1,
                MethodComponent.Builder.builder(parameterKey1)
                        .addParameter(parameterKey1, new IntegerParameter(1, v -> v > 0))
                        .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(methodComponentContext,
                methodComponentMap);

        // Invalid type
        expectThrows(ValidationException.class, () -> parameter.validate(17));
        expectThrows(ValidationException.class, () -> parameter.validate("invalid-value"));

        // Invalid value
        String invalidMethodComponentName = "invalid-method";
        MethodComponentContext invalidMethodComponentContext1 = new MethodComponentContext(invalidMethodComponentName,
                defaultParameterMap);
        expectThrows(ValidationException.class, () -> parameter.validate(invalidMethodComponentContext1));

        String invalidParameterKey = "invalid-parameter";
        Map<String, Object> invalidParameterMap1 = ImmutableMap.of(invalidParameterKey, parameterValue1);
        MethodComponentContext invalidMethodComponentContext2 = new MethodComponentContext(methodComponentName1,
                invalidParameterMap1);
        expectThrows(ValidationException.class, () -> parameter.validate(invalidMethodComponentContext2));

        String invalidParameterValue = "invalid-value";
        Map<String, Object> invalidParameterMap2 = ImmutableMap.of(parameterKey1, invalidParameterValue);
        MethodComponentContext invalidMethodComponentContext3 = new MethodComponentContext(methodComponentName1,
                invalidParameterMap2);
        expectThrows(ValidationException.class, () -> parameter.validate(invalidMethodComponentContext3));

        // valid value
        parameter.validate(methodComponentContext);
    }

    public void testMethodComponentContextParameter_getMethodComponent() {
        String methodComponentName1 = "method-1";
        String parameterKey1 = "parameter_key_1";
        Integer parameterValue1 = 12;

        Map<String, Object> defaultParameterMap = ImmutableMap.of(parameterKey1, parameterValue1);
        MethodComponentContext methodComponentContext = new MethodComponentContext(methodComponentName1,
                defaultParameterMap);

        Map<String, MethodComponent> methodComponentMap = ImmutableMap.of(
                methodComponentName1,
                MethodComponent.Builder.builder(parameterKey1)
                        .addParameter(parameterKey1, new IntegerParameter(1, v -> v > 0))
                        .build()
        );

        final MethodComponentContextParameter parameter = new MethodComponentContextParameter(methodComponentContext,
                methodComponentMap);

        // Test when method component is available
        assertEquals(methodComponentMap.get(methodComponentName1), parameter.getMethodComponent(methodComponentName1));

        // test when method component is not available
        String invalidMethod = "invalid-method";
        assertNull(parameter.getMethodComponent(invalidMethod));
    }
}
