/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010Codec;

import static org.apache.lucene.codecs.lucene101.Lucene101Codec.Mode.BEST_COMPRESSION;

public class DerivedSourceStoredFieldsFormatTests extends KNNTestCase {

    @SneakyThrows
    public void testCustomCodecDelegate() {
        // TODO: We need to replace this with a custom codec so that we can properly test. See
        // https://github.com/opensearch-project/custom-codecs/blob/main/src/main/java/org/opensearch/index/codec/customcodecs/Lucene912CustomCodec.java#L37
        Codec codec = new KNN10010Codec(new Lucene101Codec(BEST_COMPRESSION), KNN10010Codec.DEFAULT_KNN_VECTOR_FORMAT, null);

        Directory dir = newFSDirectory(createTempDir());
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(codec);

        String fieldName = "test";
        String randomString = randomAlphaOfLengthBetween(100, 1000);
        TextField basicTextField = new TextField(fieldName, randomString, Field.Store.YES);
        try (RandomIndexWriter writer = new RandomIndexWriter(random(), dir, iwc)) {
            Document doc = new Document();
            doc.add(basicTextField);
            writer.addDocument(doc);
        }

        try (IndexReader indexReader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(indexReader);
            Document doc = searcher.storedFields().document(0);
            assertEquals(randomString, doc.get(fieldName));
        }
        dir.close();
    }
}
