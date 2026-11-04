/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.io.IOException;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.store.Directory;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorDVLeafFieldData;
import org.opensearch.knn.index.KNNVectorDocValueFormat;
import org.opensearch.knn.index.VectorDataType;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KnnVectorValuesFetcherTests extends KNNTestCase {
    public void testFetchersUseWrappedFieldView() throws IOException {
        float[] vector = new float[] { 1.0f, 2.0f };
        try (Directory directory = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
                Document document = new Document();
                document.add(new KnnFloatVectorField("vector", vector));
                writer.addDocument(document);
            }
            try (DirectoryReader raw = DirectoryReader.open(directory)) {
                LeafReader leaf = raw.leaves().get(0).reader();
                KNNVectorFieldType type = mock(KNNVectorFieldType.class);
                when(type.name()).thenReturn("vector");
                KnnVectorValuesFetcher fetcher = new KnnVectorValuesFetcher(type, "vector");
                assertArrayEquals(vector, (float[]) fetcher.fetch(leaf, 0).get(0), 0.0f);

                leaf.incRef();
                try (FilterLeafReader wrapped = new FilterLeafReader(leaf) {
                    @Override
                    public FieldInfos getFieldInfos() {
                        return FieldInfos.EMPTY;
                    }

                    @Override
                    public CacheHelper getCoreCacheHelper() {
                        return null;
                    }

                    @Override
                    public CacheHelper getReaderCacheHelper() {
                        return null;
                    }
                }) {
                    assertTrue(fetcher.fetch(wrapped, 0).isEmpty());
                    var values = new KNNVectorDVLeafFieldData(wrapped, "vector", VectorDataType.FLOAT).getLeafValueFetcher(
                        KNNVectorDocValueFormat.ARRAY_FORMAT
                    );
                    assertFalse(values.advanceExact(0));
                    assertEquals(0, values.docValueCount());
                    assertArrayEquals(vector, (float[]) fetcher.fetch(leaf, 0).get(0), 0.0f);
                }
            }
        }
    }
}
