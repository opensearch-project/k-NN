/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.MethodComponent;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.Parameter;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class FaissTests extends KNNTestCase {

    public void testMethodAsMapBuilder() throws IOException {
        String methodName = "test-method";
        String methodDescription = "test-description";
        String parameter1 = "test-parameter-1";
        Integer value1 = 10;
        Integer defaultValue1 = 1;
        String parameter2 = "test-parameter-2";
        Integer value2 = 15;
        Integer defaultValue2 = 2;
        String parameter3 = "test-parameter-3";
        Integer defaultValue3 = 3;
        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter(parameter1, new Parameter.IntegerParameter(parameter1, defaultValue1, value -> value > 0))
            .addParameter(parameter2, new Parameter.IntegerParameter(parameter2, defaultValue2, value -> value > 0))
            .addParameter(parameter3, new Parameter.IntegerParameter(parameter3, defaultValue3, value -> value > 0))
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field(parameter1, value1)
            .field(parameter2, value2)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext methodComponentContext = MethodComponentContext.parse(in);

        Map<String, Object> expectedParametersMap = new HashMap<>(methodComponentContext.getParameters());
        expectedParametersMap.put(parameter3, defaultValue3);
        expectedParametersMap.remove(parameter1);
        Map<String, Object> expectedMap = new HashMap<>();
        expectedMap.put(PARAMETERS, expectedParametersMap);
        expectedMap.put(NAME, methodName);
        expectedMap.put(INDEX_DESCRIPTION_PARAMETER, methodDescription + value1);

        Map<String, Object> methodAsMap = Faiss.MethodAsMapBuilder.builder(methodDescription, methodComponent, methodComponentContext)
            .addParameter(parameter1, "", "")
            .build();

        assertEquals(expectedMap, methodAsMap);
    }

}
