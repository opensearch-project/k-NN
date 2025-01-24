/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

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
            .addParameter(paramKey, new Parameter.IntegerParameter(paramKey, 1, (v, context) -> v > 0))
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
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject();
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .dimension(1)
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .build();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext1 = MethodComponentContext.parse(in);

        MethodComponent methodComponent1 = MethodComponent.Builder.builder(methodName)
            .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
            .build();
        assertNotNull(methodComponent1.validate(componentContext1, knnMethodConfigContext));

        // Invalid parameter type
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field("valid", "invalid")
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext2 = MethodComponentContext.parse(in);

        MethodComponent methodComponent2 = MethodComponent.Builder.builder(methodName)
            .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
            .addParameter("valid", new Parameter.IntegerParameter("valid", 1, (v, context) -> v > 0))
            .build();
        assertNotNull(methodComponent2.validate(componentContext2, knnMethodConfigContext));

        // valid configuration
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field("valid1", 16)
            .field("valid2", 128)
            .endObject()
            .endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext3 = MethodComponentContext.parse(in);

        MethodComponent methodComponent3 = MethodComponent.Builder.builder(methodName)
            .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
            .addParameter("valid1", new Parameter.IntegerParameter("valid1", 1, (v, context) -> v > 0))
            .addParameter("valid2", new Parameter.IntegerParameter("valid2", 1, (v, context) -> v > 0))
            .build();
        assertNull(methodComponent3.validate(componentContext3, knnMethodConfigContext));

        // valid configuration - empty parameters
        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, methodName).endObject();
        in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext componentContext4 = MethodComponentContext.parse(in);

        MethodComponent methodComponent4 = MethodComponent.Builder.builder(methodName)
            .addSupportedDataTypes(Set.of(VectorDataType.FLOAT))
            .addParameter("valid1", new Parameter.IntegerParameter("valid1", 1, (v, context) -> v > 0))
            .addParameter("valid2", new Parameter.IntegerParameter("valid2", 1, (v, context) -> v > 0))
            .build();
        assertNull(methodComponent4.validate(componentContext4, knnMethodConfigContext));
    }

    @SuppressWarnings("unchecked")
    public void testGetAsMap_withoutGenerator() throws IOException {
        String methodName = "test-method";
        String parameterName1 = "valid1";
        String parameterName2 = "valid2";
        int default1 = 4;
        int default2 = 5;

        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter(parameterName1, new Parameter.IntegerParameter(parameterName1, default1, (v, context) -> v > 0))
            .addParameter(parameterName2, new Parameter.IntegerParameter(parameterName2, default2, (v, context) -> v > 0))
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, methodName)
            .startObject(PARAMETERS)
            .field(parameterName1, 16)
            .field(parameterName2, 128)
            .endObject()
            .endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext methodComponentContext = MethodComponentContext.parse(in);

        assertEquals(
            in,
            methodComponent.getKNNLibraryIndexingContext(
                methodComponentContext,
                KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build()
            ).getLibraryParameters()
        );

        xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, methodName).endObject();
        in = xContentBuilderToMap(xContentBuilder);
        methodComponentContext = MethodComponentContext.parse(in);

        KNNLibraryIndexingContext methodAsMap = methodComponent.getKNNLibraryIndexingContext(
            methodComponentContext,
            KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build()
        );
        assertEquals(default1, ((Map<String, Object>) methodAsMap.getLibraryParameters().get(PARAMETERS)).get(parameterName1));
        assertEquals(default2, ((Map<String, Object>) methodAsMap.getLibraryParameters().get(PARAMETERS)).get(parameterName2));
    }

    public void testGetAsMap_withGenerator() throws IOException {
        String methodName = "test-method";
        Map<String, Object> generatedMap = ImmutableMap.of("test-key", "test-value");
        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter("valid1", new Parameter.IntegerParameter("valid1", 1, (v, context) -> v > 0))
            .addParameter("valid2", new Parameter.IntegerParameter("valid2", 1, (v, context) -> v > 0))
            .setKnnLibraryIndexingContextGenerator(
                (methodComponent1, methodComponentContext, knnMethodConfigContext) -> KNNLibraryIndexingContextImpl.builder()
                    .parameters(generatedMap)
                    .build()
            )
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().field(NAME, methodName).endObject();
        Map<String, Object> in = xContentBuilderToMap(xContentBuilder);
        MethodComponentContext methodComponentContext = MethodComponentContext.parse(in);

        assertEquals(
            generatedMap,
            methodComponent.getKNNLibraryIndexingContext(
                methodComponentContext,
                KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build()
            ).getLibraryParameters()
        );
    }

    public void testBuilder() {
        String name = "test";
        MethodComponent.Builder builder = MethodComponent.Builder.builder(name);
        MethodComponent methodComponent = builder.build();

        assertEquals(0, methodComponent.getParameters().size());
        assertEquals(name, methodComponent.getName());

        builder.addParameter("test", new Parameter.IntegerParameter("test", 1, (v, context) -> v > 0));
        methodComponent = builder.build();

        assertEquals(1, methodComponent.getParameters().size());

        Map<String, Object> generatedMap = ImmutableMap.of("test-key", "test-value");
        builder.setKnnLibraryIndexingContextGenerator(
            (methodComponent1, methodComponentContext, knnMethodConfigContext) -> KNNLibraryIndexingContextImpl.builder()
                .parameters(generatedMap)
                .build()
        );
        methodComponent = builder.build();

        assertEquals(
            generatedMap,
            methodComponent.getKNNLibraryIndexingContext(null, KNNMethodConfigContext.builder().versionCreated(Version.CURRENT).build())
                .getLibraryParameters()
        );
    }

    /**
     * Test the new flow where EF_SEARCH and EF_CONSTRUCTION are set for ON_DISK mode
     * with binary quantization compression levels.
     */
    public void testGetParameterMapWithDefaultsAdded_forOnDiskWithBinaryQuantization() {
        // Set up MethodComponent and context
        String methodName = "test-method";
        String parameterEFSearch = "ef_search";
        String parameterEFConstruction = "ef_construction";

        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter(parameterEFSearch, new Parameter.IntegerParameter(parameterEFSearch, 512, (v, context) -> v > 0))
            .addParameter(parameterEFConstruction, new Parameter.IntegerParameter(parameterEFConstruction, 512, (v, context) -> v > 0))
            .build();

        // Simulate ON_DISK mode and binary quantization compression levels
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .mode(Mode.ON_DISK)  // ON_DISK mode
            .compressionLevel(CompressionLevel.x32)  // Binary quantization compression level
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(methodName, Map.of());

        // Retrieve parameter map with defaults added
        Map<String, Object> resultMap = MethodComponent.getParameterMapWithDefaultsAdded(
            methodComponentContext,
            methodComponent,
            knnMethodConfigContext
        );

        // Check that binary quantization values are used
        assertEquals(IndexHyperParametersUtil.getBinaryQuantizationEFSearchValue(), resultMap.get(parameterEFSearch));
        assertEquals(IndexHyperParametersUtil.getBinaryQuantizationEFConstructionValue(), resultMap.get(parameterEFConstruction));
    }

    public void testGetParameterMapWithDefaultsAdded_forOnDiskWithByteQuantization() {
        // Set up MethodComponent and context
        String methodName = "test-method";
        String parameterEFSearch = "ef_search";
        String parameterEFConstruction = "ef_construction";

        MethodComponent methodComponent = MethodComponent.Builder.builder(methodName)
            .addParameter(parameterEFSearch, new Parameter.IntegerParameter(parameterEFSearch, 512, (v, context) -> v > 0))
            .addParameter(parameterEFConstruction, new Parameter.IntegerParameter(parameterEFConstruction, 512, (v, context) -> v > 0))
            .build();

        // Simulate ON_DISK mode and byte quantization compression level
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .mode(Mode.ON_DISK)  // ON_DISK mode
            .compressionLevel(CompressionLevel.x4)  // Byte quantization compression level
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(methodName, Map.of());

        // Retrieve parameter map with defaults added
        Map<String, Object> resultMap = MethodComponent.getParameterMapWithDefaultsAdded(
            methodComponentContext,
            methodComponent,
            knnMethodConfigContext
        );

        // Check that byte quantization values are used
        assertEquals(IndexHyperParametersUtil.getBinaryQuantizationEFSearchValue(), resultMap.get(parameterEFSearch));
        assertEquals(IndexHyperParametersUtil.getBinaryQuantizationEFConstructionValue(), resultMap.get(parameterEFConstruction));
    }

}
