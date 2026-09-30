/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.SneakyThrows;
import org.opensearch.Version;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.io.IOException;

import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_OVERSAMPLE_PARAMETER;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

public class RescoreParserTests extends KNNTestCase {

    @SneakyThrows
    public void testStreams() {
        RescoreContext rescoreContext = RescoreContext.builder().oversampleFactor(RescoreContext.DEFAULT_OVERSAMPLE_FACTOR).build();
        validateStreams(rescoreContext);
        validateStreams(null);
        validateStreams(RescoreContext.EXPLICITLY_DISABLED_RESCORE_CONTEXT);
    }

    @SneakyThrows
    public void testStreams_withOlderVersion() {
        // Simulate a node running a version that doesn't support RESCORE_ENABLED_PARAMETER
        RescoreContext rescoreContext = RescoreContext.builder()
            .oversampleFactor(RescoreContext.DEFAULT_OVERSAMPLE_FACTOR)
            .rescoreEnabled(false)
            .build();

        try (BytesStreamOutput output = new BytesStreamOutput()) {
            output.setVersion(Version.V_3_6_0);
            RescoreParser.streamOutput(output, rescoreContext);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                in.setVersion(Version.V_3_6_0);
                RescoreContext parsedRescoreContext = RescoreParser.streamInput(in);
                // Older version doesn't serialize rescoreEnabled, so it defaults to true
                assertEquals(RescoreContext.DEFAULT_OVERSAMPLE_FACTOR, parsedRescoreContext.getOversampleFactor(), 0.0001);
                assertTrue(parsedRescoreContext.isRescoreEnabled());
            }
        }
    }

    private void validateStreams(RescoreContext rescoreContext) throws IOException {
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            RescoreParser.streamOutput(output, rescoreContext);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                RescoreContext parsedRescoreContext = RescoreParser.streamInput(in);
                assertEquals(rescoreContext, parsedRescoreContext);
            }
        }
    }

    @SneakyThrows
    public void testDoXContent() {
        float oversample = RescoreContext.MAX_OVERSAMPLE_FACTOR - 1;
        XContentBuilder expectedBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(RESCORE_PARAMETER)
            .field(RESCORE_OVERSAMPLE_PARAMETER, oversample)
            .endObject()
            .endObject();

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        RescoreParser.doXContent(builder, RescoreContext.builder().oversampleFactor(oversample).build());
        builder.endObject();
        assertEquals(expectedBuilder.toString(), builder.toString());
    }

    @SneakyThrows
    public void testFromXContent_whenValid_thenSucceed() {
        float oversample1 = RescoreContext.MAX_OVERSAMPLE_FACTOR - 1;
        XContentBuilder builder1 = XContentFactory.jsonBuilder().startObject().field(RESCORE_OVERSAMPLE_PARAMETER, oversample1).endObject();
        validateOversample(oversample1, builder1);
        XContentBuilder builder2 = XContentFactory.jsonBuilder().startObject().endObject();
        validateOversample(RescoreContext.DEFAULT_OVERSAMPLE_FACTOR, builder2);
    }

    @SneakyThrows
    public void testFromXContent_whenInvalid_thenFail() {
        XContentBuilder invalidParamBuilder = XContentFactory.jsonBuilder().startObject().field("invalid", 0).endObject();
        expectValidationException(invalidParamBuilder);

        XContentBuilder invalidParamValueBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(RESCORE_OVERSAMPLE_PARAMETER, "c")
            .endObject();
        expectValidationException(invalidParamValueBuilder);

        XContentBuilder extraParamBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(RESCORE_OVERSAMPLE_PARAMETER, RescoreContext.MAX_OVERSAMPLE_FACTOR - 1)
            .field("invalid", 0)
            .endObject();
        expectValidationException(extraParamBuilder);
    }

    private void validateOversample(float expectedOversample, XContentBuilder builder) throws IOException {
        XContentParser parser = createParser(builder);
        RescoreContext rescoreContext = RescoreParser.fromXContent(parser);
        assertEquals(expectedOversample, rescoreContext.getOversampleFactor(), 0.0001);
    }

    private void expectValidationException(XContentBuilder builder) throws IOException {
        XContentParser parser = createParser(builder);
        expectThrows(IllegalArgumentException.class, () -> RescoreParser.fromXContent(parser));
    }

    // ---- late-interaction rescore payload ----

    @SneakyThrows
    public void testStreams_lateInteraction() {
        RescoreContext ctx = RescoreContext.builder()
            .oversampleFactor(2.0f)
            .lateInteractionField("tokens")
            .lateInteractionQueryVectors(new float[][] { { 0.1f, 0.2f, 0.3f }, { 0.4f, 0.5f, 0.6f } })
            .lateInteractionSimilarity("maxSimDotProduct")
            .build();
        validateStreams(ctx);
    }

    @SneakyThrows
    public void testDoXContent_lateInteraction() {
        RescoreContext ctx = RescoreContext.builder()
            .oversampleFactor(3.0f)
            .lateInteractionField("tokens")
            .lateInteractionQueryVectors(new float[][] { { 1.0f, 2.0f } })
            .lateInteractionSimilarity("maxSimDotProduct")
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        RescoreParser.doXContent(builder, ctx);
        builder.endObject();

        // Dynamic field-name key (RFC #3439): the field name is the key, with a nested `vector`.
        String json = builder.toString();
        assertTrue(json.contains("\"tokens\""));
        assertTrue(json.contains(RescoreParser.LATE_INTERACTION_VECTOR_PARAMETER));
        assertTrue(json.contains("maxSimDotProduct"));
    }

    @SneakyThrows
    public void testFromXContent_lateInteraction_thenSucceed() {
        // "rescore": { "oversample_factor": 2, "tokens": { "vector": [[..],[..]], "similarity": "maxSimDotProduct" } }
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(RESCORE_OVERSAMPLE_PARAMETER, 2.0f)
            .startObject("tokens")
            .startArray(RescoreParser.LATE_INTERACTION_VECTOR_PARAMETER)
            .startArray()
            .value(0.1f)
            .value(0.2f)
            .value(0.3f)
            .endArray()
            .startArray()
            .value(0.4f)
            .value(0.5f)
            .value(0.6f)
            .endArray()
            .endArray()
            .field(RescoreParser.LATE_INTERACTION_SIMILARITY_PARAMETER, "maxSimDotProduct")
            .endObject()
            .endObject();

        XContentParser parser = createParser(builder);
        RescoreContext ctx = RescoreParser.fromXContent(parser);

        assertTrue(ctx.isLateInteraction());
        assertEquals("tokens", ctx.getLateInteractionField());
        assertEquals("maxSimDotProduct", ctx.getLateInteractionSimilarity());
        assertEquals(2, ctx.getLateInteractionQueryVectors().length);
        assertArrayEquals(new float[] { 0.1f, 0.2f, 0.3f }, ctx.getLateInteractionQueryVectors()[0], 1e-6f);
        assertArrayEquals(new float[] { 0.4f, 0.5f, 0.6f }, ctx.getLateInteractionQueryVectors()[1], 1e-6f);
    }

    @SneakyThrows
    public void testFromXContent_lateInteraction_defaultsOversample() {
        // No oversample_factor given with a late-interaction rescore -> defaults to 3.0.
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("tokens")
            .startArray(RescoreParser.LATE_INTERACTION_VECTOR_PARAMETER)
            .startArray()
            .value(0.1f)
            .value(0.2f)
            .endArray()
            .endArray()
            .endObject()
            .endObject();
        XContentParser parser = createParser(builder);
        RescoreContext ctx = RescoreParser.fromXContent(parser);
        assertEquals(RescoreParser.LATE_INTERACTION_DEFAULT_OVERSAMPLE_FACTOR, ctx.getOversampleFactor(), 1e-6f);
    }

    @SneakyThrows
    public void testFromXContent_lateInteraction_missingVector_thenFail() {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("tokens")
            .field(RescoreParser.LATE_INTERACTION_SIMILARITY_PARAMETER, "maxSimDotProduct")
            .endObject()
            .endObject();
        XContentParser parser = createParser(builder);
        expectThrows(IllegalArgumentException.class, () -> RescoreParser.fromXContent(parser));
    }
}
