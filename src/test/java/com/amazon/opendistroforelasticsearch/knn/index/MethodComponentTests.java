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
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;

import java.io.IOException;
import java.util.Map;

import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.NAME;
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.PARAMETERS;

public class MethodComponentTests extends KNNTestCase {
    /**
     * Test name getter
     */
    public void testGetName() {
        String name = "test";
        MethodComponent methodComponent = MethodComponent.Builder.builder(name).build();
        assertEquals(name, methodComponent.getName());
    }

    /**
     * Test parameter getter
     */
    public void testGetParameters() {
        String name = "test";
        String paramKey = "key";
        MethodComponent methodComponent = MethodComponent.Builder.builder(name)
                .addParameter(paramKey, new Parameter.IntegerParameter(1, v -> v > 0))
                .build();
        assertEquals(1, methodComponent.getParameters().size());
        assertTrue(methodComponent.getParameters().containsKey(paramKey));
    }

    /**
     * Test validation
     */
    public void testValidate() throws IOException {
        // Invalid parameter key
        String methodName = "test-method";
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .startObject(PARAMETERS)
                .field("invalid", "invalid")
                .endObject()
                .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext1 = MethodComponentContext.parse(in);

        MethodComponent methodComponent1 = MethodComponent.Builder.builder(methodName).build();

        expectThrows(ValidationException.class, () -> methodComponent1.validate(componentContext1));

        // Invalid parameter type
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .startObject(PARAMETERS)
                .field("valid", "invalid")
                .endObject()
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext2 = MethodComponentContext.parse(in);

        MethodComponent methodComponent2 = MethodComponent.Builder.builder(methodName)
                .addParameter("valid", new Parameter.IntegerParameter(1, v -> v > 0))
                .build();

        expectThrows(ValidationException.class, () -> methodComponent2.validate(componentContext2));

        // valid configuration
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .startObject(PARAMETERS)
                .field("valid1", 16)
                .field("valid2", 128)
                .endObject()
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext3 = MethodComponentContext.parse(in);

        MethodComponent methodComponent3 = MethodComponent.Builder.builder(methodName)
                .addParameter("valid1", new Parameter.IntegerParameter(1, v -> v > 0))
                .addParameter("valid2", new Parameter.IntegerParameter(1, v -> v > 0))
                .build();
        methodComponent3.validate(componentContext3);

        // valid configuration - empty parameters
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext4 = MethodComponentContext.parse(in);

        MethodComponent methodComponent4 = MethodComponent.Builder.builder(methodName)
                .addParameter("valid1", new Parameter.IntegerParameter(1, v -> v > 0))
                .addParameter("valid2", new Parameter.IntegerParameter(1, v -> v > 0))
                .build();
        methodComponent4.validate(componentContext4);
    }

    public void testBuilder() {
        String name = "test";
        MethodComponent.Builder builder = MethodComponent.Builder.builder(name);
        MethodComponent methodComponent = builder.build();

        assertEquals(0, methodComponent.getParameters().size());
        assertEquals(name, methodComponent.getName());

        builder.addParameter("test", new Parameter.IntegerParameter(1, v -> v > 0));
        methodComponent = builder.build();

        assertEquals(1, methodComponent.getParameters().size());
    }
}
