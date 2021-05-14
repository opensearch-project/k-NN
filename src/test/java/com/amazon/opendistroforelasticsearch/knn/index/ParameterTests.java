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

package com.amazon.opendistroforelasticsearch.knn.index;

import com.amazon.opendistroforelasticsearch.knn.KNNTestCase;
import org.opensearch.common.ValidationException;

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
        final Parameter.IntegerParameter parameter = new Parameter.IntegerParameter(1,
                v -> v > 0);

        // Invalid type
        expectThrows(ValidationException.class, () -> parameter.validate("String"));

        // Invalid value
        expectThrows(ValidationException.class, () -> parameter.validate(-1));

        // valid value
        parameter.validate(12);
    }
}
