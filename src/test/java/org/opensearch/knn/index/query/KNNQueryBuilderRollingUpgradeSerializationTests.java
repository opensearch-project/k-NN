/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.Version;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.util.KNNClusterUtil;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

/**
 * Regression tests for issue #1622 (flaky BWC failure: {@code unexpected byte [0x05]}).
 *
 * <p><b>Root cause (fixed):</b> {@link org.opensearch.knn.index.query.parser.KNNQueryBuilderParser} used to decide
 * which optional fields go on the transport wire using the <i>cluster</i> minimum version
 * ({@code IndexUtil.isClusterOnOrAfterMinRequiredVersion} -&gt; {@code KNNClusterUtil.getClusterMinVersion()})
 * instead of the <i>transport stream</i> version ({@code StreamOutput#getVersion()} / {@code StreamInput#getVersion()}).
 *
 * <p>Cluster minimum version is derived from cluster state, which propagates <i>eventually</i>. During a rolling
 * upgrade two communicating nodes could transiently hold different views of it, so the writer and the reader made
 * different decisions about whether an optional field was present. The byte stream then desynced, and a later
 * {@code readBoolean} in {@code SearchSourceBuilder} landed on a stray byte -&gt; 0x05.
 *
 * <p>The fix (see {@link KNNQueryBuilder#doWriteTo} and the {@code StreamInput} constructor) gates serialization on
 * the stream version instead, matching the pattern already used by
 * {@link org.opensearch.knn.index.query.parser.RescoreParser}. The stream version is the min of the two nodes for a
 * given connection and is therefore identical on both ends, so writer and reader can no longer disagree.
 *
 * <p>These tests hold the stream version fixed and vary only the mocked cluster-min-version view to prove that the
 * mocked view (previously the culprit) no longer has any effect on wire format or round-trip correctness.
 */
public class KNNQueryBuilderRollingUpgradeSerializationTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final float[] QUERY_VECTOR = { 1.0f, 2.0f, 3.0f, 4.0f };
    private static final int K = 5;

    // expandNested is gated on EXPAND_NESTED == Version.V_2_19_0 (see IndexUtil). It is also the LAST field
    // written by KNNQueryBuilderParser.streamOutput, so a writer/reader disagreement about it produces a clean,
    // isolated trailing-byte desync -- which made it a good field to characterize the original bug with.
    private static final Version NEW_NODE_VIEW = Version.V_2_19_0; // believes cluster supports expand_nested
    private static final Version OLD_NODE_VIEW = Version.V_2_18_0; // believes it does not

    // Stream version at/after the expand_nested threshold: the feature should always be included.
    private static final Version STREAM_VERSION_SUPPORTS_FEATURE = Version.V_2_19_0;
    // Stream version before the threshold: the feature should always be omitted.
    private static final Version STREAM_VERSION_BELOW_FEATURE = Version.V_2_18_0;

    private NamedWriteableRegistry registry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, KNNQueryBuilder.NAME, KNNQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    /** Points KNNClusterUtil.getClusterMinVersion() at the given version (what a node currently "believes"). */
    private void setClusterMinVersionView(final Version version) {
        final ClusterService clusterService = mockClusterService(version);
        KNNClusterUtil.instance().initialize(clusterService, mock(IndexNameExpressionResolver.class));
    }

    private KNNQueryBuilder queryWithExpandNested() {
        return KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).k(K).expandNested(Boolean.TRUE).build();
    }

    private BytesReference serialize(final KNNQueryBuilder query, final Version streamVersion) throws Exception {
        try (BytesStreamOutput out = new BytesStreamOutput()) {
            out.setVersion(streamVersion);
            out.writeNamedWriteable(query);
            return out.bytes();
        }
    }

    /**
     * Proves the root cause is fixed: with the stream version held constant, changing only the mocked cluster-min
     * version view no longer changes the serialized byte length. Before the fix, this produced two DIFFERENT
     * lengths for the same object over the same stream version -- wire format must depend only on the stream.
     */
    public void testWireFormatDependsOnlyOnStreamVersion_notClusterMinVersion() throws Exception {
        final KNNQueryBuilder query = queryWithExpandNested();

        setClusterMinVersionView(NEW_NODE_VIEW);
        final int lengthWithNewNodeView = serialize(query, STREAM_VERSION_SUPPORTS_FEATURE).length();

        setClusterMinVersionView(OLD_NODE_VIEW);
        final int lengthWithOldNodeView = serialize(query, STREAM_VERSION_SUPPORTS_FEATURE).length();

        assertEquals(
            "Wire format must depend only on the stream version, not on the mutable cluster-min version view",
            lengthWithNewNodeView,
            lengthWithOldNodeView
        );
    }

    /**
     * Proves the desync is fixed: a "new" node writes the query while it happens to believe cluster-min is 2.19,
     * and an "old" node reads it while it happens to believe cluster-min is 2.18 -- over the SAME stream version
     * that supports the feature. Both nodes must now agree on the wire format (based on the stream), so the field
     * round-trips correctly and no bytes are left unconsumed.
     */
    public void testRoundTrip_isStableAcrossClusterMinVersionViews() throws Exception {
        final KNNQueryBuilder query = queryWithExpandNested();

        setClusterMinVersionView(NEW_NODE_VIEW);
        final BytesReference bytes = serialize(query, STREAM_VERSION_SUPPORTS_FEATURE);

        setClusterMinVersionView(OLD_NODE_VIEW);
        try (StreamInput in = new NamedWriteableAwareStreamInput(bytes.streamInput(), registry())) {
            in.setVersion(STREAM_VERSION_SUPPORTS_FEATURE);
            final QueryBuilder deserialized = in.readNamedWriteable(QueryBuilder.class);

            assertTrue(deserialized instanceof KNNQueryBuilder);
            assertEquals(Boolean.TRUE, ((KNNQueryBuilder) deserialized).getExpandNested());
            assertEquals("No bytes should be left unconsumed regardless of cluster-min view", 0, in.available());
        }
    }

    /**
     * Symmetric case: when the stream version is BELOW the feature threshold, the field must be consistently
     * omitted by both nodes regardless of their cluster-min view -- proving genuine backwards compatibility with
     * older nodes still works correctly after the fix.
     */
    public void testRoundTrip_whenStreamVersionBelowFeatureThreshold_fieldOmittedConsistently() throws Exception {
        final KNNQueryBuilder query = queryWithExpandNested();

        setClusterMinVersionView(NEW_NODE_VIEW);
        final BytesReference bytes = serialize(query, STREAM_VERSION_BELOW_FEATURE);

        setClusterMinVersionView(OLD_NODE_VIEW);
        try (StreamInput in = new NamedWriteableAwareStreamInput(bytes.streamInput(), registry())) {
            in.setVersion(STREAM_VERSION_BELOW_FEATURE);
            final QueryBuilder deserialized = in.readNamedWriteable(QueryBuilder.class);

            assertTrue(deserialized instanceof KNNQueryBuilder);
            assertNull(
                "Below the feature threshold, expand_nested must not be on the wire regardless of cluster-min view",
                ((KNNQueryBuilder) deserialized).getExpandNested()
            );
            assertEquals("No bytes should be left unconsumed regardless of cluster-min view", 0, in.available());
        }
    }
}
