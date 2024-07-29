/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.plugins.SearchPlugin;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Benchmarks for impact of changes around query parsing
 */
@Warmup(iterations = 5, time = 10)
@Measurement(iterations = 3, time = 10)
@Fork(3)
@State(Scope.Benchmark)
public class QueryParsingBenchmarks {
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("field", "value");
    private static final NamedXContentRegistry NAMED_X_CONTENT_REGISTRY = xContentRegistry();

    @Param({ "128", "1024" })
    private int dimension;
    @Param({ "basic", "filter" })
    private String type;

    private BytesReference bytesReference;

    @Setup
    public void setup() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.startObject("test");
        builder.field(KNNQueryBuilder.VECTOR_FIELD.getPreferredName(), generateVectorWithOnes(dimension));
        builder.field(KNNQueryBuilder.K_FIELD.getPreferredName(), 1);
        if (type.equals("filter")) {
            builder.field(KNNQueryBuilder.FILTER_FIELD.getPreferredName(), TERM_QUERY);
        }
        builder.endObject();
        builder.endObject();
        bytesReference = BytesReference.bytes(builder);
    }

    @Benchmark
    public void fromXContent(final Blackhole bh) throws IOException {
        XContentParser xContentParser = createParser();
        bh.consume(KNNQueryBuilderParser.fromXContent(xContentParser));
    }

    private XContentParser createParser() throws IOException {
        XContentParser contentParser = createParser(bytesReference);
        contentParser.nextToken();
        return contentParser;
    }

    private float[] generateVectorWithOnes(final int dimensions) {
        float[] vector = new float[dimensions];
        Arrays.fill(vector, (float) 1);
        return vector;
    }

    private XContentParser createParser(final BytesReference data) throws IOException {
        BytesArray array = (BytesArray) data;
        return JsonXContent.jsonXContent.createParser(
            NAMED_X_CONTENT_REGISTRY,
            LoggingDeprecationHandler.INSTANCE,
            array.array(),
            array.offset(),
            array.length()
        );
    }

    private static NamedXContentRegistry xContentRegistry() {
        List<NamedXContentRegistry.Entry> list = ClusterModule.getNamedXWriteables();
        SearchPlugin.QuerySpec<?> spec = new SearchPlugin.QuerySpec<>(
            TermQueryBuilder.NAME,
            TermQueryBuilder::new,
            TermQueryBuilder::fromXContent
        );
        list.add(new NamedXContentRegistry.Entry(QueryBuilder.class, spec.getName(), (p, c) -> spec.getParser().fromXContent(p)));
        return new NamedXContentRegistry(list);
    }
}
