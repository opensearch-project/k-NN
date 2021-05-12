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
/*
 *   Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
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
import static com.amazon.opendistroforelasticsearch.knn.common.KNNConstants.SPACE_TYPE;


public class KNNMethodTests extends KNNTestCase {
    /**
     * Test KNNMethod method component getter
     */
    public void testGetMethodComponent() {
        String name = "test";
        KNNMethod knnMethod = KNNMethod.Builder.builder(MethodComponent.Builder.builder(name).build())
                .build();
        assertEquals(name, knnMethod.getMethodComponent().getName());
    }

    /**
     * Test KNNMethod has space
     */
    public void testHasSpace() {
        String name = "test";
        KNNMethod knnMethod = KNNMethod.Builder.builder(MethodComponent.Builder.builder(name).build())
                .addSpaces(SpaceType.L2, SpaceType.COSINESIMIL)
                .build();
        assertTrue(knnMethod.hasSpace(SpaceType.L2));
        assertTrue(knnMethod.hasSpace(SpaceType.COSINESIMIL));
        assertFalse(knnMethod.hasSpace(SpaceType.INNER_PRODUCT));
    }

    /**
     * Test KNNMethod validate
     */
    public void testValidate() throws IOException {
        String methodName = "test-method";
        KNNMethod knnMethod = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName).build())
                .addSpaces(SpaceType.L2)
                .build();

        // Invalid space
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .field(SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
                .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);

        expectThrows(ValidationException.class, () -> knnMethod.validate(knnMethodContext1));

        // Invalid methodComponent
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .field(SPACE_TYPE, SpaceType.L2.getValue())
                .startObject(PARAMETERS)
                .field("invalid", "invalid")
                .endObject()
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);

        expectThrows(ValidationException.class, () -> knnMethod.validate(knnMethodContext2));

        // Valid everything
        xContentBuilder = XContentFactory.jsonBuilder().startObject()
                .field(NAME, methodName)
                .field(SPACE_TYPE, SpaceType.L2.getValue())
                .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext3 = KNNMethodContext.parse(in);
        knnMethod.validate(knnMethodContext3);
    }

    public void testBuilder() {
        String name = "test";
        KNNMethod.Builder builder = KNNMethod.Builder.builder(MethodComponent.Builder.builder(name).build());
        KNNMethod knnMethod = builder.build();

        assertEquals(name, knnMethod.getMethodComponent().getName());

        builder.addSpaces(SpaceType.L2);
        knnMethod = builder.build();

        assertTrue(knnMethod.hasSpace(SpaceType.L2));
    }
}
