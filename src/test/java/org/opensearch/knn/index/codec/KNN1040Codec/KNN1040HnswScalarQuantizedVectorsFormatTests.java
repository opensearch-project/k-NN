/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1040HnswScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertNotNull(format);
        String str = format.toString();
        assertTrue(str.contains("KNN1040HnswScalarQuantizedVectorsFormat"));
        assertTrue(str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testCustomConstructor() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            32,
            200,
            1,
            null
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat();
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_invalidMaxConn_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 0, 100, 1, null)
        );
    }

    public void testConstructor_invalidBeamWidth_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 0, 1, null)
        );
    }

    public void testConstructor_singleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1040HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 100, 1, Executors.newFixedThreadPool(1))
        );
    }

    public void testConstructor_allEncodings() {
        for (ScalarEncoding encoding : ScalarEncoding.values()) {
            KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(encoding, 16, 100, 1, null);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }

    public void testConstructor_whenCustomTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            16,
            100,
            1,
            null,
            500
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_whenMaxValueTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            16,
            100,
            1,
            null,
            Integer.MAX_VALUE
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_whenZeroTinySegmentsThreshold_thenSucceeds() {
        KNN1040HnswScalarQuantizedVectorsFormat format = new KNN1040HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            16,
            100,
            1,
            null,
            0
        );
        assertNotNull(format);
        assertEquals("KNN1040HnswScalarQuantizedVectorsFormat", format.getName());
    }

    @SneakyThrows
    public void testOpenReader_thenRawAndQuantizedVectorsAdvisedForRandomAccess() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            try (
                IndexWriter writer = new IndexWriter(
                    dir,
                    newIndexWriterConfig().setCodec(new UnitTestCodec(KNN1040HnswScalarQuantizedVectorsFormat::new))
                        .setUseCompoundFile(false)
                        .setMergePolicy(newLogMergePolicy(false))
                )
            ) {
                for (int i = 0; i < 4; i++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("field", new float[] { i, i + 1, i + 2, i + 3 }, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
            }
            final Set<String> randomAccessFiles = ConcurrentHashMap.newKeySet();
            dir.setReadAdvice((name, context) -> {
                if (context.hints().contains(DataAccessHint.RANDOM)) {
                    randomAccessFiles.add(name);
                }
                return Optional.empty();
            });

            try (DirectoryReader ignored = DirectoryReader.open(dir)) {
                assertTrue(randomAccessFiles.toString(), randomAccessFiles.stream().anyMatch(name -> name.endsWith(".vec")));
                assertTrue(randomAccessFiles.toString(), randomAccessFiles.stream().anyMatch(name -> name.endsWith(".veq")));
            }
        }
    }
}
