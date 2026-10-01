/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.store.Directory;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.opensearch.common.CheckedConsumer;
import org.opensearch.knn.KNNTestCase;

public class LeafReaderUtilTests extends KNNTestCase {

    public void testTryGetSegmentReader_whenSegmentReader_thenReturnsIt() throws Exception {
        withSingleDocumentLeaf(leafReader -> assertSame(leafReader, LeafReaderUtil.tryGetSegmentReader(leafReader)));
    }

    public void testTryGetSegmentReader_whenFilterLeafReader_thenUnwraps() throws Exception {
        withSingleDocumentLeaf(leafReader -> {
            LeafReader wrapped = new FilterLeafReader(leafReader) {
                @Override
                public CacheHelper getCoreCacheHelper() {
                    return in.getCoreCacheHelper();
                }

                @Override
                public CacheHelper getReaderCacheHelper() {
                    return in.getReaderCacheHelper();
                }
            };
            assertSame(leafReader, LeafReaderUtil.tryGetSegmentReader(wrapped));
        });
    }

    public void testTryGetSegmentReader_whenFilterCodecReader_thenUnwraps() throws Exception {
        withSingleDocumentLeaf(leafReader -> {
            FilterCodecReader wrapped = new FilterCodecReader((SegmentReader) leafReader) {
                @Override
                public CacheHelper getCoreCacheHelper() {
                    return in.getCoreCacheHelper();
                }

                @Override
                public CacheHelper getReaderCacheHelper() {
                    return in.getReaderCacheHelper();
                }
            };
            assertSame(leafReader, LeafReaderUtil.tryGetSegmentReader(wrapped));
        });
    }

    public void testTryGetSegmentReader_whenMemoryIndexReader_thenReturnsNull() {
        assertNull(LeafReaderUtil.tryGetSegmentReader(memoryIndexLeaf()));
    }

    public void testLeafReaderName_whenSegment_thenUsesSegmentName() throws Exception {
        withSingleDocumentLeaf(
            leafReader -> assertEquals(((SegmentReader) leafReader).getSegmentName(), LeafReaderUtil.leafReaderName(leafReader))
        );
    }

    public void testLeafReaderName_whenNoSegment_thenUsesClassName() {
        LeafReader leafReader = memoryIndexLeaf();
        assertEquals(leafReader.getClass().getSimpleName(), LeafReaderUtil.leafReaderName(leafReader));
    }

    private LeafReader memoryIndexLeaf() {
        return new MemoryIndex().createSearcher().getIndexReader().leaves().get(0).reader();
    }

    private void withSingleDocumentLeaf(final CheckedConsumer<LeafReader, Exception> consumer) throws Exception {
        try (Directory directory = newDirectory(); IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
            writer.addDocument(new Document());
            writer.commit();
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                consumer.accept(reader.leaves().get(0).reader());
            }
        }
    }
}
