/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import com.google.common.collect.ImmutableMap;
import org.opensearch.common.ValidationException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.*;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;

public class AbstractKNNLibraryTests extends KNNTestCase {

    public void testGetVersion() {
        String testVersion = "test-version";
        TestAbstractKNNLibrary testAbstractKNNLibrary = new TestAbstractKNNLibrary(Collections.emptyMap(), testVersion);
        assertEquals(testVersion, testAbstractKNNLibrary.getVersion());
    }

    public void testGetMethod() {
        String methodName1 = "test-method-1";
        KNNMethod knnMethod1 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName1).build()).build();

        String methodName2 = "test-method-2";
        KNNMethod knnMethod2 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName2).build()).build();

        Map<String, KNNMethod> knnMethodMap = ImmutableMap.of(methodName1, knnMethod1, methodName2, knnMethod2);

        TestAbstractKNNLibrary testAbstractKNNLibrary = new TestAbstractKNNLibrary(knnMethodMap, "");
        assertEquals(knnMethod1, testAbstractKNNLibrary.getMethod(methodName1));
        assertEquals(knnMethod2, testAbstractKNNLibrary.getMethod(methodName2));
        expectThrows(IllegalArgumentException.class, () -> testAbstractKNNLibrary.getMethod("invalid"));
    }

    public void testValidateMethod() throws IOException {
        // Invalid - method not supported
        String methodName1 = "test-method-1";
        KNNMethod knnMethod1 = KNNMethod.Builder.builder(MethodComponent.Builder.builder(methodName1).build()).build();

        Map<String, KNNMethod> methodMap = ImmutableMap.of(methodName1, knnMethod1);
        TestAbstractKNNLibrary testAbstractKNNLibrary1 = new TestAbstractKNNLibrary(methodMap, "");

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, "invalid").endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        expectThrows(IllegalArgumentException.class, () -> testAbstractKNNLibrary1.validateMethod(knnMethodContext1));

        // Invalid - method validation
        String methodName2 = "test-method-2";
        KNNMethod knnMethod2 = new KNNMethod(MethodComponent.Builder.builder(methodName2).build(), Collections.emptySet()) {
            @Override
            public ValidationException validate(KNNMethodContext knnMethodContext) {
                return new ValidationException();
            }
        };

        methodMap = ImmutableMap.of(methodName2, knnMethod2);
        TestAbstractKNNLibrary testAbstractKNNLibrary2 = new TestAbstractKNNLibrary(methodMap, "");
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, methodName2).endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);
        assertNotNull(testAbstractKNNLibrary2.validateMethod(knnMethodContext2));
    }

    public void testEngineSpecificMethods() {
        String methodName1 = "test-method-1";
        QueryContext engineSpecificMethodContext = new QueryContext(VectorQueryType.K);
        EngineSpecificMethodContext context = ctx -> ImmutableMap.of(
            "myparameter",
            new Parameter.BooleanParameter("myparameter", null, value -> true)
        );

        TestAbstractKNNLibrary testAbstractKNNLibrary1 = new TestAbstractKNNLibrary(
            Collections.emptyMap(),
            Map.of(methodName1, context),
            ""
        );

        assertNotNull(testAbstractKNNLibrary1.getMethodContext(methodName1));
        assertTrue(
            testAbstractKNNLibrary1.getMethodContext(methodName1)
                .supportedMethodParameters(engineSpecificMethodContext)
                .containsKey("myparameter")
        );
    }

    public void testGetMethodAsMap() {
        String methodName = "test-method-1";
        SpaceType spaceType = SpaceType.DEFAULT;
        Map<String, Object> generatedMap = ImmutableMap.of("test-key", "test-param");
        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .setMapGenerator(((methodComponent1, methodComponentContext) -> generatedMap))
            .build();
        KNNMethod knnMethod = KNNMethod.Builder.builder(methodComponent).build();

        TestAbstractKNNLibrary testAbstractKNNLibrary = new TestAbstractKNNLibrary(ImmutableMap.of(methodName, knnMethod), "");

        // Check that map is expected
        Map<String, Object> expectedMap = new HashMap<>(generatedMap);
        expectedMap.put(KNNConstants.SPACE_TYPE, spaceType.getValue());
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.DEFAULT,
            spaceType,
            new MethodComponentContext(methodName, Collections.emptyMap())
        );
        assertEquals(expectedMap, testAbstractKNNLibrary.getMethodAsMap(knnMethodContext));

        // Check when invalid method is passed in
        KNNMethodContext invalidKnnMethodContext = new KNNMethodContext(
            KNNEngine.DEFAULT,
            spaceType,
            new MethodComponentContext("invalid", Collections.emptyMap())
        );
        expectThrows(IllegalArgumentException.class, () -> testAbstractKNNLibrary.getMethodAsMap(invalidKnnMethodContext));
    }

    private static class TestAbstractKNNLibrary extends AbstractKNNLibrary {
        public TestAbstractKNNLibrary(Map<String, KNNMethod> methods, String currentVersion) {
            super(methods, Collections.emptyMap(), currentVersion);
        }

        public TestAbstractKNNLibrary(
            Map<String, KNNMethod> methods,
            Map<String, EngineSpecificMethodContext> engineSpecificMethodContextMap,
            String currentVersion
        ) {
            super(methods, engineSpecificMethodContextMap, currentVersion);
        }

        @Override
        public String getExtension() {
            return null;
        }

        @Override
        public String getCompoundExtension() {
            return null;
        }

        @Override
        public float score(float rawScore, SpaceType spaceType) {
            return 0;
        }

        @Override
        public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
            return 0f;
        }

        public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
            return 0f;
        }

        @Override
        public int estimateOverheadInKB(KNNMethodContext knnMethodContext, int dimension) {
            return 0;
        }

        @Override
        public Boolean isInitialized() {
            return null;
        }

        @Override
        public void setInitialized(Boolean isInitialized) {

        }
    }
}
