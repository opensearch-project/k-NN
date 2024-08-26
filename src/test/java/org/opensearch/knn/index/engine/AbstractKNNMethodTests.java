/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;

public class AbstractKNNMethodTests extends KNNTestCase {

    private static class TestKNNMethod extends AbstractKNNMethod {
        public TestKNNMethod(MethodComponent methodComponent, Set<SpaceType> spaces, KNNLibrarySearchContext engineSpecificMethodContext) {
            super(methodComponent, spaces, engineSpecificMethodContext);
        }
    }

    /**
     * Test KNNMethod has space
     */
    public void testHasSpace() {
        String name = "test";
        KNNMethod knnMethod = new TestKNNMethod(
            MethodComponent.Builder.builder(name).build(),
            Set.of(SpaceType.L2, SpaceType.COSINESIMIL),
            EMPTY_ENGINE_SPECIFIC_CONTEXT
        );
        assertTrue(knnMethod.isSpaceTypeSupported(SpaceType.L2));
        assertTrue(knnMethod.isSpaceTypeSupported(SpaceType.COSINESIMIL));
        assertFalse(knnMethod.isSpaceTypeSupported(SpaceType.INNER_PRODUCT));
    }

    /**
     * Test KNNMethod validate
     */
    public void testValidate() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        String methodName = "test-method";
        KNNMethod knnMethod = new TestKNNMethod(
            MethodComponent.Builder.builder(methodName).addSupportedDataTypes(Set.of(VectorDataType.FLOAT)).build(),
            Set.of(SpaceType.L2),
            EMPTY_ENGINE_SPECIFIC_CONTEXT
        );

        // Invalid space
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        assertNotNull(knnMethod.validate(knnMethodContext1, knnMethodConfigContext));

        // Invalid methodComponent
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);

        assertNotNull(knnMethod.validate(knnMethodContext2, knnMethodConfigContext));

        // Valid everything
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext3 = KNNMethodContext.parse(in);
        assertNull(knnMethod.validate(knnMethodContext3, knnMethodConfigContext));
    }

    /**
     * Test KNNMethod validateWithData
     */
    public void testValidateWithContext() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        String methodName = "test-method";
        KNNMethod knnMethod = new TestKNNMethod(
            MethodComponent.Builder.builder(methodName).addSupportedDataTypes(Set.of(VectorDataType.FLOAT)).build(),
            Set.of(SpaceType.L2),
            EMPTY_ENGINE_SPECIFIC_CONTEXT
        );

        // Invalid space
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.INNER_PRODUCT.getValue())
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        assertNotNull(knnMethod.validate(knnMethodContext1, knnMethodConfigContext));

        // Invalid methodComponent
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);
        assertNotNull(knnMethod.validate(knnMethodContext2, knnMethodConfigContext));

        // Valid everything
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext3 = KNNMethodContext.parse(in);
        assertNull(knnMethod.validate(knnMethodContext3, knnMethodConfigContext));
    }

    public void testGetKNNLibraryIndexingContext() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(4)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        SpaceType spaceType = SpaceType.DEFAULT;
        String methodName = "test-method";
        Map<String, Object> generatedMap = new HashMap<>(ImmutableMap.of("test-key", "test-value"));
        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .setKnnLibraryIndexingContextGenerator(
                ((methodComponent1, methodComponentContext, methodConfigContext) -> KNNLibraryIndexingContextImpl.builder()
                    .parameters(methodComponentContext.getParameters())
                    .build())
            )
            .build();

        KNNMethod knnMethod = new TestKNNMethod(methodComponent, Set.of(SpaceType.L2), EMPTY_ENGINE_SPECIFIC_CONTEXT);

        Map<String, Object> expectedMap = new HashMap<>(generatedMap);
        expectedMap.put(KNNConstants.SPACE_TYPE, spaceType.getValue());
        expectedMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());

        assertEquals(
            expectedMap,
            knnMethod.getKNNLibraryIndexingContext(
                new KNNMethodContext(KNNEngine.DEFAULT, spaceType, new MethodComponentContext(methodName, generatedMap)),
                knnMethodConfigContext
            ).getLibraryParameters()
        );
    }

    public void testGetKNNLibrarySearchContext() {
        String methodName = "test-method";
        KNNLibrarySearchContext knnLibrarySearchContext = new DefaultHnswSearchContext();
        KNNMethod knnMethod = new TestKNNMethod(
            MethodComponent.Builder.builder(methodName).build(),
            Set.of(SpaceType.L2),
            knnLibrarySearchContext
        );
        assertEquals(knnLibrarySearchContext, knnMethod.getKNNLibrarySearchContext());
    }
}
