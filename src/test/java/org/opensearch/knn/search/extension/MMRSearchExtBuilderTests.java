/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.extension;

import org.junit.Before;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.ParsingException;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.search.processor.mmr.MMRTestCase;
import org.opensearch.search.pipeline.SearchPipelineService;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MMRSearchExtBuilderTests extends MMRTestCase {
    private float DELTA = 1e-6F;
    private float DEFAULT_DIVERSITY = 0.5f;
    private ClusterService clusterService = mock(ClusterService.class);

    @Before
    public void setUp() throws Exception {
        super.setUp();
        SearchPipelineService searchPipelineService = mock(SearchPipelineService.class);
        when(searchPipelineService.isSystemGeneratedFactoryEnabled(any())).thenReturn(true);
        KNNClusterUtil.instance().initialize(clusterService, mock(IndexNameExpressionResolver.class));
        KNNClusterUtil.instance().setSearchPipelineService(searchPipelineService);
    }

    public void testBuilderDefaultsAndValues() {
        MMRSearchExtBuilder builder = new MMRSearchExtBuilder.Builder().candidates(10)
            .vectorFieldPath("vec")
            .spaceType("l2")
            .vectorFieldDataType("float")
            .build();

        assertEquals(DEFAULT_DIVERSITY, builder.getDiversity(), DELTA);
        assertEquals(10, (int) builder.getCandidates());
        assertEquals("vec", builder.getVectorFieldPath());
        assertEquals(SpaceType.L2, builder.getSpaceType());
        assertEquals(VectorDataType.FLOAT, builder.getVectorFieldDataType());
    }

    public void testBuilder_whenNegativeDiversity_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        builder.diversity(-0.1f);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, builder::build);
        String expectedError = "diversity in mmr query extension must be between 0.0 and 1.0";
        assertEquals(expectedError, ex.getMessage());
    }

    public void testBuilder_whenLargeDiversity_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        builder.diversity(1.1f);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, builder::build);
        String expectedError = "diversity in mmr query extension must be between 0.0 and 1.0";
        assertEquals(expectedError, ex.getMessage());
    }

    public void testBuilder_whenNegativeCandidates_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        builder.candidates(-1);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, builder::build);
        String expectedError = "candidates in mmr query extension must be larger than 0.";
        assertEquals(expectedError, ex.getMessage());
    }

    public void testBuilder_whenInvalidSpaceType_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> builder.spaceType("invalid"));
        String expectedError = "vector_field_space_type in mmr query extension is not valid";
        assertTrue(ex.getMessage().contains(expectedError));
    }

    public void testBuilder_whenInvalidVectorDataType_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> builder.vectorFieldDataType("invalid"));
        String expectedError = "vector_field_data_type in mmr query extension is not valid";
        assertTrue(ex.getMessage().contains(expectedError));
    }

    public void testBuilder_whenEmptyVectorPath_thenException() {
        MMRSearchExtBuilder.Builder builder = new MMRSearchExtBuilder.Builder();
        builder.vectorFieldPath("");
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, builder::build);
        String expectedError = "vector_field_path in mmr query extension should not be an empty string.";
        assertEquals(expectedError, ex.getMessage());
    }

    public void testEqualsAndHashCode() {
        MMRSearchExtBuilder builder1 = new MMRSearchExtBuilder.Builder().diversity(0.7f)
            .candidates(3)
            .vectorFieldPath("path")
            .spaceType("l2")
            .build();
        MMRSearchExtBuilder builder2 = new MMRSearchExtBuilder.Builder().diversity(0.7f)
            .candidates(3)
            .vectorFieldPath("path")
            .spaceType("l2")
            .build();
        MMRSearchExtBuilder builder3 = new MMRSearchExtBuilder.Builder().diversity(0.9f).build();

        assertEquals(builder1, builder2);
        assertEquals(builder1.hashCode(), builder2.hashCode());
        assertNotEquals(builder1, builder3);
    }

    public void testSerializationRoundTrip() throws IOException {
        MMRSearchExtBuilder original = new MMRSearchExtBuilder.Builder().diversity(0.6f)
            .candidates(20)
            .vectorFieldPath("vec")
            .spaceType("l2")
            .vectorFieldDataType("byte")
            .build();

        BytesStreamOutput out = new BytesStreamOutput();
        original.writeTo(out);

        MMRSearchExtBuilder deserialized = new MMRSearchExtBuilder(out.bytes().streamInput());

        assertEquals(original, deserialized);
    }

    public void testToXContentAndParse() throws IOException {
        MMRSearchExtBuilder original = new MMRSearchExtBuilder.Builder().diversity(0.9f)
            .candidates(15)
            .vectorFieldPath("vector")
            .spaceType(SpaceType.COSINESIMIL.getValue())
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        original.toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
        xContentBuilder.endObject();

        try (XContentParser parser = createParser(xContentBuilder)) {
            parser.nextToken(); // start object
            parser.nextToken(); // field name "mmr"
            parser.nextToken(); // start object
            MMRSearchExtBuilder parsed = MMRSearchExtBuilder.parse(parser);

            assertEquals(original, parsed);
        }
    }

    public void testParse_whenUnsupportedField_thenException() throws IOException {
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.startObject("mmr");
        xContentBuilder.field("unsupported_field", "value");
        xContentBuilder.endObject();
        xContentBuilder.endObject();

        try (XContentParser parser = createParser(xContentBuilder)) {
            parser.nextToken(); // start object
            parser.nextToken(); // field name "mmr"
            parser.nextToken(); // start object
            ParsingException ex = assertThrows(ParsingException.class, () -> MMRSearchExtBuilder.parse(parser));
            String expectedError = "[mmr] query extension does not support [unsupported_field]";
            assertEquals(expectedError, ex.getMessage());
        }
    }

    public void testParse_whenMMRProcessorsNotEnabled_thenException() throws IOException {
        SearchPipelineService searchPipelineService = mock(SearchPipelineService.class);
        when(searchPipelineService.isSystemGeneratedFactoryEnabled(any())).thenReturn(false);
        KNNClusterUtil.instance().initialize(clusterService, mock(IndexNameExpressionResolver.class));
        KNNClusterUtil.instance().setSearchPipelineService(searchPipelineService);

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
        xContentBuilder.startObject();
        xContentBuilder.startObject("mmr");
        xContentBuilder.endObject();
        xContentBuilder.endObject();

        try (XContentParser parser = createParser(xContentBuilder)) {
            parser.nextToken(); // start object
            parser.nextToken(); // field name "mmr"
            parser.nextToken(); // start object
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> MMRSearchExtBuilder.parse(parser));
            String expectedError =
                "We need to enable [mmr_over_sample_factory, mmr_rerank_factory] in the cluster setting [cluster.search.enabled_system_generated_factories] to support the mmr search extension.";
            assertEquals(expectedError, ex.getMessage());
        }
    }
}
