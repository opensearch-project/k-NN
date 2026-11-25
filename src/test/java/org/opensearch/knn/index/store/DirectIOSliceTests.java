/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.opensearch.action.admin.indices.forcemerge.ForceMergeRequest;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Which input a field's vectors come from in a real node, whose directory chain, shard path and segment reader no
 * hand-built unit test reaches. Each case asks the way Lucene's memory-segment scorer and k-NN's {@code WarmupUtil}
 * do, through {@code ((HasIndexSlice) values).getSlice()}. The base class starts a node per test, so each case checks
 * several fields or indices where that reads naturally.
 */
public class DirectIOSliceTests extends KNNSingleNodeTestCase {

    // 8,192 floats is 32,768 bytes, so 321 documents is 10,518,528 bytes, just over the 10 MiB floor.
    private static final int DIMENSION = 8192;
    private static final int DOC_COUNT = 321;

    private static final String FIELD = "target_field";
    private static final String SECOND_FIELD = "second_field";
    private static final String LUCENE_FIELD = "lucene_field";

    @Override
    public void setUp() throws Exception {
        super.setUp();
        DirectIOTestUtils.assumeDirectIOSupported();
    }

    public void testAnIndexThatIsOnGivesADirectIOSlice() throws Exception {
        final IndexService indexService = createEligibleIndex("direct-io-slice-on", true, KNNEngine.FAISS);
        assertTrue(
            "a faiss .vec over the floor in a non-compound segment of an index that is on must be read with O_DIRECT",
            readsWithDirectIO(indexService, FIELD)
        );
    }

    public void testAnIndexThatIsOffGivesTheOriginalInputsSlice() throws Exception {
        final IndexService indexService = createEligibleIndex("direct-io-slice-off", false, KNNEngine.FAISS);
        // readsWithDirectIO also asserts that a slice which is not a Direct I/O reader is a MemorySegmentAccessInput.
        assertFalse("an index that is off must read as main does", readsWithDirectIO(indexService, FIELD));
    }

    public void testTheFlipReachesTheOpenSegmentBothWays() throws Exception {
        final IndexService indexService = createEligibleIndex("direct-io-slice-flip", false, KNNEngine.FAISS);
        // The same open segment throughout: no refresh, no close and open, only the settings update.
        assertFalse(readsWithDirectIO(indexService, FIELD));
        setDirectIO("direct-io-slice-flip", true);
        assertTrue("the flip to on reaches a segment that is already open", readsWithDirectIO(indexService, FIELD));
        setDirectIO("direct-io-slice-flip", false);
        assertFalse("and the flip back to off stops it", readsWithDirectIO(indexService, FIELD));
    }

    public void testTwoIndicesOneOnOneOffAndTheWildcardPut() throws Exception {
        final IndexService on = createEligibleIndex("direct-io-slice-pair-on", true, KNNEngine.FAISS);
        final IndexService off = createEligibleIndex("direct-io-slice-pair-off", false, KNNEngine.FAISS);
        assertTrue("the index that is on reads with O_DIRECT", readsWithDirectIO(on, FIELD));
        assertFalse("the index that is off does not", readsWithDirectIO(off, FIELD));

        // The rollback the design asks for: one wildcard settings update turns every index off.
        setDirectIO("*", false);
        assertFalse("the wildcard put turns the on index off", readsWithDirectIO(on, FIELD));
        assertFalse(readsWithDirectIO(off, FIELD));
    }

    public void testASegmentThatFailsACheckAtOpenGivesNoDirectIOSlice() throws Exception {
        // Not force merged, so the segments are compound at the shipped defaults and each is under the floor. Which
        // check a given segment fails is not fixed, so KNNDirectIODirectoryTests isolates them one at a time.
        final IndexService flushed = createIndexWithVectors("direct-io-slice-flushed", true, KNNEngine.FAISS, DIMENSION, DOC_COUNT, false);
        assertTrue("the fixture must be more than one flushed segment", leafCount(flushed) > 1);
        assertFalse("a flushed segment must not be read with O_DIRECT", readsWithDirectIO(flushed, FIELD));

        // One merged segment, so not compound, but 16 by 100 floats is 6,400 bytes: only the floor can fail here.
        final IndexService small = createIndexWithVectors("direct-io-slice-small", true, KNNEngine.FAISS, 16, 100, true);
        assertFalse("a file under the floor must not be read with O_DIRECT", readsWithDirectIO(small, FIELD));
    }

    public void testOnlyFaissFieldsGetADirectIOSlice() throws Exception {
        // One segment, three fields: a Lucene engine field, whose format never hands its files to the delegate, and
        // two faiss fields, which both use it. The choice is per file inside one segment.
        final String index = "direct-io-slice-mixed";
        final IndexService mixed = createKnnIndexService(index, true);
        createKnnMapping(index, LUCENE_FIELD, DIMENSION, KNNEngine.LUCENE);
        createKnnMapping(index, FIELD, DIMENSION, KNNEngine.FAISS);
        createKnnMapping(index, SECOND_FIELD, DIMENSION, KNNEngine.FAISS);
        indexVectors(index, DIMENSION, DOC_COUNT, LUCENE_FIELD, FIELD, SECOND_FIELD);
        forceMergeToOneSegment(index);

        assertOneNonCompoundSegment(mixed);
        assertFalse("a Lucene engine .vec is never offered to the delegate", readsWithDirectIO(mixed, LUCENE_FIELD));
        assertTrue("the first faiss field is read with O_DIRECT", readsWithDirectIO(mixed, FIELD));
        assertTrue("and so is the second", readsWithDirectIO(mixed, SECOND_FIELD));
    }

    public void testStillUsesDirectIOAfterCloseAndOpen() throws Exception {
        final String index = "direct-io-slice-reopen";
        final IndexService indexService = createEligibleIndex(index, true, KNNEngine.FAISS);
        assertTrue(readsWithDirectIO(indexService, FIELD));

        client().admin().indices().prepareClose(index).get();
        client().admin().indices().prepareOpen(index).get();
        ensureGreen(index);

        final IndexService reopened = getInstanceFromNode(org.opensearch.indices.IndicesService.class).indexService(resolveIndex(index));
        assertNotNull("the index must be open again", reopened);
        assertTrue("the reopened segment is read with O_DIRECT again", readsWithDirectIO(reopened, FIELD));
    }

    // The slice never leaves the searcher that owns it: the check happens inside the try block and only the verdict
    // is returned.
    private boolean readsWithDirectIO(final IndexService indexService, final String field) throws IOException {
        final IndexShard shard = indexService.iterator().next();
        try (Engine.Searcher searcher = shard.acquireSearcher("direct-io-slice-test")) {
            for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
                final FloatVectorValues values = leaf.reader().getFloatVectorValues(field);
                if (values == null) {
                    continue;
                }
                assertTrue(
                    "the flat vector values of " + field + " must expose their slice, as Lucene's scorer needs",
                    values instanceof HasIndexSlice
                );
                final IndexInput slice = ((HasIndexSlice) values).getSlice();
                assertNotNull("the slice of " + field + " must not be null", slice);
                if (slice instanceof DirectIOVectorIndexInput) {
                    return true;
                }
                assertTrue(
                    "a slice that is not a Direct I/O reader must be the original input's own, not " + slice.getClass(),
                    slice instanceof MemorySegmentAccessInput
                );
                return false;
            }
        }
        throw new AssertionError("no leaf of this index has the field " + field);
    }

    private int leafCount(final IndexService indexService) {
        final IndexShard shard = indexService.iterator().next();
        try (Engine.Searcher searcher = shard.acquireSearcher("direct-io-slice-leaves")) {
            return searcher.getIndexReader().leaves().size();
        }
    }

    private void assertOneNonCompoundSegment(final IndexService indexService) {
        final IndexShard shard = indexService.iterator().next();
        try (Engine.Searcher searcher = shard.acquireSearcher("direct-io-slice-layout")) {
            final List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
            assertEquals("the fixture must be one merged segment", 1, leaves.size());
            assertFalse(
                "the merged segment must not be compound, or its .vec is declined at open",
                Lucene.segmentReader(leaves.get(0).reader()).getSegmentInfo().info.getUseCompoundFile()
            );
        }
    }

    private IndexService createEligibleIndex(final String index, final boolean directIO, final KNNEngine engine) throws Exception {
        return createIndexWithVectors(index, directIO, engine, DIMENSION, DOC_COUNT, true);
    }

    private IndexService createIndexWithVectors(
        final String index,
        final boolean directIO,
        final KNNEngine engine,
        final int dimension,
        final int docCount,
        final boolean forceMerge
    ) throws Exception {
        final IndexService indexService = createKnnIndexService(index, directIO);
        createKnnMapping(index, FIELD, dimension, engine);
        indexVectors(index, dimension, docCount, FIELD);
        if (forceMerge) {
            forceMergeToOneSegment(index);
            assertOneNonCompoundSegment(indexService);
        }
        return indexService;
    }

    private IndexService createKnnIndexService(final String index, final boolean directIO) {
        final Settings settings = Settings.builder()
            .put(getKNNDefaultIndexSettingsBuildsGraphAlways())
            .put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, directIO)
            .build();
        return createIndex(index, settings);
    }

    private void createKnnMapping(final String index, final String field, final int dimension, final KNNEngine engine) throws IOException {
        // The base class overload that puts the engine in the method and leaves the space type at its default of l2.
        createKnnIndexMapping(index, field, dimension, engine);
    }

    // One bulk request per 64 documents, a flush halfway and one refresh at the end. The flush keeps the documents in
    // two segments at least, because a force merge leaves a lone flushed segment as it is: compound.
    private void indexVectors(final String index, final int dimension, final int docCount, final String... fields) throws Exception {
        final Random random = new Random(1_234L);
        final int chunk = 64;
        for (int from = 0; from < docCount; from += chunk) {
            final int to = Math.min(from + chunk, docCount);
            final BulkRequest bulk = new BulkRequest();
            for (int doc = from; doc < to; doc++) {
                final Float[] vector = new Float[dimension];
                for (int i = 0; i < dimension; i++) {
                    vector[i] = random.nextFloat();
                }
                final XContentBuilder source = XContentFactory.jsonBuilder().startObject();
                for (final String field : fields) {
                    source.field(field, vector);
                }
                source.endObject();
                bulk.add(new IndexRequest(index).id(Integer.toString(doc)).source(source));
            }
            bulk.setRefreshPolicy(WriteRequest.RefreshPolicy.NONE);
            final BulkResponse response = client().bulk(bulk).get();
            assertFalse(response.buildFailureMessage(), response.hasFailures());
            if (from < docCount / 2 && to >= docCount / 2) {
                client().admin().indices().prepareFlush(index).get();
            }
        }
        client().admin().indices().prepareRefresh(index).get();
    }

    // One merged segment is also a non-compound segment: it is the whole index against a no-CFS ratio of 0.1.
    private void forceMergeToOneSegment(final String index) {
        client().admin().indices().forceMerge(new ForceMergeRequest(index).maxNumSegments(1)).actionGet();
        client().admin().indices().prepareRefresh(index).get();
    }

    private void setDirectIO(final String index, final boolean enabled) {
        updateIndexSetting(index, Settings.builder().put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled).build());
    }
}
