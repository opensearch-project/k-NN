/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.common.ValidationException;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.*;
import org.opensearch.knn.index.engine.model.QueryContext;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.NAME;

public class AbstractKNNLibraryTests extends KNNTestCase {

    private final static String CURRENT_VERSION = "test-version";
    private final static String INVALID_METHOD_THROWS_VALIDATION_NAME = "test-method-1";
    private final static KNNMethod INVALID_METHOD_THROWS_VALIDATION = new AbstractKNNMethod(
        MethodComponent.Builder.builder(INVALID_METHOD_THROWS_VALIDATION_NAME).addSupportedDataTypes(Set.of(VectorDataType.FLOAT)).build(),
        Set.of(SpaceType.DEFAULT),
        new DefaultHnswSearchContext()
    ) {
        @Override
        public ValidationException validate(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            return new ValidationException();
        }
    };
    private final static String VALID_METHOD_NAME = "test-method-2";
    private final static KNNLibrarySearchContext VALID_METHOD_CONTEXT = ctx -> ImmutableMap.of(
        "myparameter",
        new Parameter.BooleanParameter("myparameter", null, (v, context) -> true)
    );
    private final static Map<String, Object> VALID_EXPECTED_MAP = ImmutableMap.of("test-key", "test-param");
    private final static KNNMethod VALID_METHOD = new AbstractKNNMethod(
        MethodComponent.Builder.builder(VALID_METHOD_NAME)
            .setKnnLibraryIndexingContextGenerator(
                (methodComponent, methodComponentContext, knnMethodConfigContext) -> KNNLibraryIndexingContextImpl.builder()
                    .parameters(new HashMap<>(VALID_EXPECTED_MAP))
                    .build()
            )
            .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
            .build(),
        Set.of(SpaceType.DEFAULT),
        VALID_METHOD_CONTEXT
    ) {
    };
    private final static AbstractKNNLibrary TEST_LIBRARY = new TestAbstractKNNLibrary(
        ImmutableMap.of(INVALID_METHOD_THROWS_VALIDATION_NAME, INVALID_METHOD_THROWS_VALIDATION, VALID_METHOD_NAME, VALID_METHOD),
        CURRENT_VERSION
    );

    public void testGetVersion() {
        assertEquals(CURRENT_VERSION, TEST_LIBRARY.getVersion());
    }

    public void testValidateMethod() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        // Invalid - method not supported
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, "invalid").endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext1 = KNNMethodContext.parse(in);
        assertNotNull(TEST_LIBRARY.validateMethod(knnMethodContext1, knnMethodConfigContext));

        // Invalid - method validation
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, INVALID_METHOD_THROWS_VALIDATION_NAME).endObject();
        in = xContentBuilderToMap(xContentBuilder);
        KNNMethodContext knnMethodContext2 = KNNMethodContext.parse(in);
        expectThrows(IllegalStateException.class, () -> TEST_LIBRARY.validateMethod(knnMethodContext2, knnMethodConfigContext));
    }

    public void testEngineSpecificMethods() {
        QueryContext engineSpecificMethodContext = new QueryContext(VectorQueryType.K);
        assertNotNull(TEST_LIBRARY.getKNNLibrarySearchContext(VALID_METHOD_NAME));
        assertTrue(
            TEST_LIBRARY.getKNNLibrarySearchContext(VALID_METHOD_NAME)
                .supportedMethodParameters(engineSpecificMethodContext)
                .containsKey("myparameter")
        );
    }

    public void testGetKNNLibraryIndexingContext() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(org.opensearch.Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        // Check that map is expected
        Map<String, Object> expectedMap = new HashMap<>(VALID_EXPECTED_MAP);
        expectedMap.put(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue());
        expectedMap.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.DEFAULT,
            SpaceType.DEFAULT,
            new MethodComponentContext(VALID_METHOD_NAME, Collections.emptyMap())
        );
        assertEquals(
            expectedMap,
            TEST_LIBRARY.getKNNLibraryIndexingContext(knnMethodContext, knnMethodConfigContext).getLibraryParameters()
        );

        // Check when invalid method is passed in
        KNNMethodContext invalidKnnMethodContext = new KNNMethodContext(
            KNNEngine.DEFAULT,
            SpaceType.DEFAULT,
            new MethodComponentContext("invalid", Collections.emptyMap())
        );
        expectThrows(
            IllegalArgumentException.class,
            () -> TEST_LIBRARY.getKNNLibraryIndexingContext(invalidKnnMethodContext, knnMethodConfigContext)
        );
    }

    private static class TestAbstractKNNLibrary extends AbstractKNNLibrary {
        public TestAbstractKNNLibrary(Map<String, KNNMethod> methods, String currentVersion) {
            super(methods, currentVersion);
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
        public int estimateOverheadInKB(KNNMethodContext knnMethodContext, KNNMethodConfigContext knnMethodConfigContext) {
            return 0;
        }

        @Override
        public Boolean isInitialized() {
            return null;
        }

        @Override
        public void setInitialized(Boolean isInitialized) {

        }

        @Override
        public ResolvedMethodContext resolveMethod(
            KNNMethodContext knnMethodContext,
            KNNMethodConfigContext knnMethodConfigContext,
            boolean shouldRequireTraining,
            SpaceType spaceType
        ) {
            return null;
        }
    }
}
