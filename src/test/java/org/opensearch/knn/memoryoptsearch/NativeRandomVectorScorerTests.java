/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.junit.Test;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class NativeRandomVectorScorerTests extends KNNTestCase {
    @Test
    public void fp16MaxIPTest() {
        // Single MemorySegment scenario
        doFp16ScoringTest(
            SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT,
            KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            false
        );

        // Multi MemorySegments scenario
        doFp16ScoringTest(
            SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT,
            KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            true
        );
    }

    @Test
    public void fp16MaxL2Test() {
        // Single MemorySegment scenario
        doFp16ScoringTest(SimdVectorComputeService.SimilarityFunctionType.FP16_L2, KNNVectorSimilarityFunction.EUCLIDEAN, false);

        // Multi MemorySegments scenario
        doFp16ScoringTest(SimdVectorComputeService.SimilarityFunctionType.FP16_L2, KNNVectorSimilarityFunction.EUCLIDEAN, true);
    }

    @SneakyThrows
    private void doFp16ScoringTest(
        final SimdVectorComputeService.SimilarityFunctionType functionType,
        final KNNVectorSimilarityFunction similarityFunction,
        final boolean multiSegmentsScenario
    ) {

        // Create repo
        final Path tempDirPath = createTempDir();

        // Write 10 vectors
        final int dimension = 123;
        final int numVectors = 100;
        final Path tempFile = Paths.get(tempDirPath.toFile().getAbsolutePath(), "test.bin");
        final long flatVectorStartOffset = 1555;
        final List<float[]> vectors = new ArrayList<>();
        try (
            FileChannel channel = FileChannel.open(
                tempFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING
            )
        ) {
            // Write few bytes before flat vector section
            final ByteBuffer dummyBytesBuffer = ByteBuffer.allocate((int) flatVectorStartOffset);
            for (int i = 0; i < flatVectorStartOffset; ++i) {
                dummyBytesBuffer.put((byte) i);
            }
            dummyBytesBuffer.flip();
            channel.write(dummyBytesBuffer);

            final ByteBuffer byteBuffer = ByteBuffer.allocate(numVectors * dimension * Short.BYTES).order(ByteOrder.nativeOrder());
            final ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
            for (int i = 0; i < numVectors; ++i) {
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; ++j) {
                    final float floatValue = ThreadLocalRandom.current().nextFloat();
                    final short fp16Value = Float.floatToFloat16(floatValue);
                    shortBuffer.put(fp16Value);
                    vector[j] = Float.float16ToFloat(fp16Value);
                }
                vectors.add(vector);
            }
            channel.write(byteBuffer);
        }

        // Make a query vector
        float[] queryVec = new float[dimension];
        for (int j = 0; j < dimension; ++j) {
            queryVec[j] = Float.float16ToFloat(Float.floatToFloat16(ThreadLocalRandom.current().nextFloat()));
        }

        // Test vector scoring
        final MMapDirectory mmapDirectory;
        if (multiSegmentsScenario) {
            // Use 1024 bytes as chunk size for mmap.
            // Since we have 26155 = 1555 (dummy bytes) + 123 * 2 * 100 bytes,
            // so the expected number of MemorySegment should be 26 = Ceil(26155 / 1024) = Ceil(25.54).
            mmapDirectory = new MMapDirectory(tempDirPath, 1024);
        } else {
            mmapDirectory = new MMapDirectory(tempDirPath);
        }

        try (mmapDirectory) {
            try (final IndexInput indexInput = mmapDirectory.openInput(tempFile.getFileName().toString(), IOContext.DEFAULT)) {
                // Extract mmap pointer
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                    indexInput,
                    flatVectorStartOffset,
                    Files.size(tempFile) - flatVectorStartOffset
                );

                if (multiSegmentsScenario) {
                    // Since the start offset of flat vector is 1555, the first chunk should be excluded. Therefore, 25 = 26 - 1
                    assertEquals(25, addressAndSize.length / 2);
                    // the first part size is 2 * 1024 - 1555
                    // <------> => 2 * 1024 - 1555 = 493
                    // 2048
                    // |-------------------|------------x------|
                    // 0 1024 ^--------- 1555
                    assertEquals(493, addressAndSize[1]);
                } else {
                    assertEquals(1, addressAndSize.length / 2);

                    // 24600 = 2 (=FP16 byte size) * 123 (=dimension) * 100 (=#vectors)
                    assertEquals(24600, addressAndSize[1]);
                }

                // Save search context first
                SimdVectorComputeService.saveSearchContext(queryVec, addressAndSize, functionType.ordinal());

                // Test single vector scoring
                for (int i = 0; i < numVectors; ++i) {
                    final float score = SimdVectorComputeService.scoreSimilarity(i);
                    final float expectedScore = similarityFunction.compare(queryVec, vectors.get(i));
                    assertEquals(expectedScore, score, 1e-3);
                }

                // Test bulk scoring
                final int batchSize = 10;
                for (int i = 0; i < numVectors; i += batchSize) {
                    int[] vectorIds = java.util.stream.IntStream.rangeClosed(i, i + batchSize).toArray();
                    float[] scores = new float[batchSize];
                    SimdVectorComputeService.scoreSimilarityInBulk(vectorIds, scores, batchSize);

                    for (int j = 0; j < batchSize; j++) {
                        final float expectedScore = similarityFunction.compare(queryVec, vectors.get(vectorIds[j]));
                        assertEquals(expectedScore, scores[j], 1e-3);
                    }
                }
            }
        }
    }
}
