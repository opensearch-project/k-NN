/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.generate.IndexingType;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsReader;
import org.opensearch.knn.index.warmup.WarmableReader;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Warmup-only tests for {@link Faiss1040ScalarQuantizedKnnVectorsReader}.
 * Ingests real vectors using {@link Faiss1040ScalarQuantizedKnnVectorsFormat}, triggers warmup via
 * {@link WarmableReader#warmUp(String)}, and verifies that {@code .faiss}, {@code .veb}, and {@code .vec}
 * files are fully read. No search queries are performed.
 *
 * // Feature: warmup-delegation-tests, Property 4: Faiss SQ warmup reads all three file types
 */
public class Faiss1040SQWarmupTests extends KNNTestCase {

    private static final int DIMENSIONS = 16;
    private static final int TOTAL_NUM_DOCS = 300;
    private static final String FIELD_NAME = "target_field";
    private static final String PARAMETERS_JSON = "{"
        + "\"index_description\":\"BHNSW16,Flat\","
        + "\"spaceType\":\"innerproduct\","
        + "\"name\":\"hnsw\","
        + "\"data_type\":\"float\","
        + "\"parameters\":{"
        + "\"ef_search\":256,"
        + "\"ef_construction\":256,"
        + "\"m\":16,"
        + "\"encoder\":{\"name\":\"sq\",\"bits\":1}"
        + "}"
        + "}";

    @SneakyThrows
    public void testWarmup() {
        final List<Integer> documentIds = IndexingType.DENSE.generateDocumentIds(TOTAL_NUM_DOCS);
        final int maxDoc = documentIds.isEmpty() ? 0 : documentIds.getLast() + 1;
        final float[][] vectors = generateVectorsForDocs(documentIds, DIMENSIONS);

        final Path tempDir = createTempDir("faiss1040sq_warmup");
        try (Directory rawDirectory = new MMapDirectory(tempDir)) {
            // Write vectors using the real Faiss1040 SQ format (creates .faiss, .veb, .vec)
            final FieldInfo fieldInfo = createFieldInfo();
            final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

            // Prepare segment info
            final byte[] segId = StringHelper.randomId();
            final SegmentInfo segInfo = new SegmentInfo(
                rawDirectory,
                Version.LATEST,
                Version.LATEST,
                "_0",
                maxDoc,
                false,
                false,
                null,
                Collections.emptyMap(),
                segId,
                Collections.emptyMap(),
                null
            );
            final SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.NO_OUTPUT,
                rawDirectory,
                segInfo,
                fieldInfos,
                null,
                IOContext.DEFAULT
            );

            // Prepare format
            final Faiss1040ScalarQuantizedKnnVectorsFormat format = new Faiss1040ScalarQuantizedKnnVectorsFormat();

            // Write vectors (quantizing + build a graph)
            try (KnnVectorsWriter writer = format.fieldsWriter(writeState)) {
                @SuppressWarnings("unchecked")
                KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
                for (int i = 0; i < documentIds.size(); i++) {
                    fw.addValue(documentIds.get(i), vectors[i]);
                }
                writer.flush(maxDoc, null);
                writer.finish();
            }

            // Set the segment files so the reader can find them
            final Set<String> segmentFiles = new HashSet<>();
            for (String file : rawDirectory.listAll()) {
                if (file.startsWith("_")) {
                    segmentFiles.add(file);
                }
            }
            segInfo.setFiles(segmentFiles);

            // Wrap directory with spy to track file reads
            final FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory spyDirectory =
                new FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory(rawDirectory);

            // Open reader through the spy directory
            final SegmentReadState readState = new SegmentReadState(spyDirectory, segInfo, fieldInfos, IOContext.DEFAULT);

            try (KnnVectorsReader reader = format.fieldsReader(readState)) {
                // Reset read flags so we only track reads during warmup
                spyDirectory.resetReadFlags();

                // Trigger warmup via WarmableReader interface
                assert reader instanceof WarmableReader;
                ((WarmableReader) reader).warmUp(FIELD_NAME);

                // Assert all three file types were read during warmup
                assertTrue("Warmup should read .faiss file", spyDirectory.wasExtensionRead(".faiss"));
                assertTrue("Warmup should read .veq file (quantized vectors)", spyDirectory.wasExtensionRead(".veq"));
                assertTrue("Warmup should read .vec file", spyDirectory.wasExtensionRead(".vec"));
            }
        }
    }

    private static FieldInfo createFieldInfo() {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(
                KNNConstants.PARAMETERS,
                PARAMETERS_JSON,
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                "float",
                KNNConstants.KNN_ENGINE,
                "faiss",
                KNNConstants.SQ_CONFIG,
                "bits=1",
                org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD,
                "true",
                KNNConstants.SPACE_TYPE,
                "innerproduct"
            ),
            0,
            0,
            0,
            DIMENSIONS,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false,
            false
        );
    }

    private static float[][] generateVectorsForDocs(List<Integer> documentIds, int dimension) {
        final Random rng = new Random(42);
        final float[][] vectors = new float[documentIds.size()][dimension];
        for (int i = 0; i < documentIds.size(); i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = rng.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }
}
