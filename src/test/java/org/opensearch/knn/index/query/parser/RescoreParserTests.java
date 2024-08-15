/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.SneakyThrows;
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
}
