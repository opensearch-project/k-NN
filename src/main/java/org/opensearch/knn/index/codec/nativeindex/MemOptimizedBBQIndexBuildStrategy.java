/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.lucene.lucene102.Lucene102BinaryFlatVectorsScorer;
import org.opensearch.lucene.lucene102.Lucene102BinaryQuantizedVectorsReader;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Memory-optimized build strategy for Faiss BBQ (Binary Quantized) HNSW indexes.
 *
 * <h2>Architecture Overview</h2>
 * This strategy builds a Faiss HNSW graph on top of 1-bit binary quantized vectors produced by
 * Lucene's {@link Lucene102BinaryQuantizedVectorsReader}. By the time this strategy is invoked,
 * Lucene has already written the quantized vectors to flat vector files (.vec and .veb) during
 * the flush/merge phase of {@code FaissBBQ990KnnVectorsWriter}. This strategy then:
 * <ol>
 *   <li>Reads back the quantized vectors and their correction factors from Lucene's on-disk format</li>
 *   <li>Transfers them to off-heap (native) memory via JNI</li>
 *   <li>Delegates to Faiss C++ to build the HNSW graph over those quantized vectors</li>
 *   <li>Writes only the HNSW graph to disk (using {@code IO_FLAG_SKIP_STORAGE}), without
 *       duplicating the flat vector storage — the quantized vectors remain in Lucene's .veb file</li>
 * </ol>
 *
 * <h2>BBQ Quantization</h2>
 * BBQ quantizes each float vector component down to 1 bit, producing a compact binary code.
 * To preserve distance accuracy, each quantized vector is accompanied by correction factors:
 * <ul>
 *   <li>{@code lowerInterval} (float) — lower bound of the quantization interval</li>
 *   <li>{@code upperInterval} (float) — upper bound of the quantization interval</li>
 *   <li>{@code additionalCorrection} (float) — residual correction term for score adjustment</li>
 *   <li>{@code quantizedComponentSum} (int) — sum of quantized component values, used in
 *       asymmetric distance computation (ADC)</li>
 * </ul>
 * These correction factors enable accurate approximate distance calculations between a 4-bit
 * quantized query and 1-bit quantized data vectors during search (see docs/5_bulk_simd_bbq_adc.md).
 *
 * <h2>Storage Separation</h2>
 * The resulting .faiss file contains only the HNSW graph structure (adjacency lists, levels, etc.)
 * with a "null" storage section placeholder. At search time, {@code FaissBBQFlatIndex} is plugged
 * in as a virtual storage layer that reads quantized vectors from Lucene's
 * {@link Lucene102BinaryQuantizedVectorsReader} instead of from the .faiss file
 * (see docs/4_faiss_mos_bbq_storage_plugin.md).
 *
 * <h2>SIMD-Accelerated Search</h2>
 * During search, the ADC distance computation between 4-bit query vectors and 1-bit data vectors
 * is offloaded to native SIMD code (AVX512 / ARM NEON) via {@code FaissBBQNativeRandomVectorScorer}
 * for maximum throughput (see docs/5_bulk_simd_bbq_adc.md).
 *
 * @see NativeIndexBuildStrategy
 * @see NativeIndexBuildStrategyFactory — returns this strategy when field info contains faiss_bbq_config
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class MemOptimizedBBQIndexBuildStrategy implements NativeIndexBuildStrategy {
    private static MemOptimizedBBQIndexBuildStrategy INSTANCE = new MemOptimizedBBQIndexBuildStrategy();

    public static MemOptimizedBBQIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    /**
     * Main entry point: builds a Faiss BBQ HNSW index and writes it to disk.
     *
     * <p>The high-level flow is:
     * <ol>
     *   <li>Initialize vector values to determine dimensionality</li>
     *   <li>Construct a {@link SegmentReadState} to read back the already-flushed .vec and .veb files
     *       containing full-precision and binary quantized vectors respectively</li>
     *   <li>Open readers for both full-precision vectors ({@link Lucene99FlatVectorsFormat}) and
     *       binary quantized vectors ({@link Lucene102BinaryQuantizedVectorsReader})</li>
     *   <li>Extract the centroid dot-product and quantized vector byte size from the binarized values,
     *       which are needed to initialize the native Faiss BBQ index structure</li>
     *   <li>Initialize the Faiss BBQ index in C++ via JNI, allocating off-heap memory for the
     *       HNSW graph and quantized vector storage</li>
     *   <li>Delegate to {@link #doBuildAndWriteIndex} for the actual data transfer and graph construction</li>
     * </ol>
     *
     * @param indexInfo contains all parameters needed for index construction, including vector values,
     *                  segment write state, field info, engine parameters, and output buffer
     * @throws IOException                if reading quantized vectors or writing the index fails
     * @throws IndexBuildAbortedException if the build is cancelled
     */
    @Override
    public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException, IndexBuildAbortedException {
        final KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        // Advance the iterator to the first document so we can read the vector dimension.
        // Without this, dimension() may return 0 since no vector has been loaded yet.
        initializeVectorValues(knnVectorValues);

        // Construct a SegmentReadState scoped to only this field so we can open readers
        // for the .vec (full-precision) and .veb (binary quantized) files that were already
        // written during the flush/merge phase by FaissBBQ990KnnVectorsWriter.
        final SegmentWriteState writeState = indexInfo.getSegmentWriteState();
        final SegmentReadState readState = new SegmentReadState(
            writeState.directory, writeState.segmentInfo,
            // Wrap only this single field — we don't need other fields' metadata
            new FieldInfos(new FieldInfo[] { indexInfo.getFieldInfo() }), writeState.context, indexInfo.getFieldInfo().getName()
        );

        // Open the full-precision flat vectors reader (.vec file) — this is required as a
        // dependency for the binary quantized reader, which needs access to the original
        // vectors for centroid computation and corrective term derivation.
        // TODO: This must come from Faiss BBQ format. Until 2_faiss_format.md is done, we hard code the format in here.
        final FlatVectorsReader fullPrecisionVectorsReader =
            new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()).fieldsReader(readState);

        // Open the binary quantized vectors reader (.veb file). This reader provides access to:
        //   - BinarizedVectorValues: the 1-bit quantized binary codes
        //   - Correction factors (lowerInterval, upperInterval, additionalCorrection, quantizedComponentSum)
        //   - Centroid dot-product: a precomputed term used in the ADC scoring formula
        // The Lucene102BinaryFlatVectorsScorer wraps the full-precision scorer to support
        // scoring with binarized representations.
        final Lucene102BinaryQuantizedVectorsReader faissBBQVectorsReader = new Lucene102BinaryQuantizedVectorsReader(
            readState,
            fullPrecisionVectorsReader,
            new Lucene102BinaryFlatVectorsScorer(fullPrecisionVectorsReader.getFlatVectorScorer())
        );

        // Retrieve the binarized vector values for this field. The returned FloatVectorValues
        // is actually a BinarizedVectorValues instance that provides both the quantized binary
        // codes and their associated correction factors.
        final FloatVectorValues floatVectorValues = faissBBQVectorsReader.getFloatVectorValues(indexInfo.getFieldInfo().getName());
        final Lucene102BinaryQuantizedVectorsReader.BinarizedVectorValues binarizedVectorValues =
            (Lucene102BinaryQuantizedVectorsReader.BinarizedVectorValues) floatVectorValues;

        // The byte length of a single quantized vector. For 1-bit quantization of D dimensions,
        // this is ceil(D/8) bytes, padded to 64-bit alignment: ((D + 63) / 64) * 64 / 8.
        final int quantizedVecBytes = binarizedVectorValues.getQuantizedVectorValues().vectorValue(0).length;

        // The centroid dot-product is a precomputed scalar used in the ADC scoring formula.
        // It appears as a correction term: score += additionalCorrection + indexCorrection - centroidDp
        // (see the quantizedScore function in docs/5_bulk_simd_bbq_adc.md).
        final float centroidDp = binarizedVectorValues.getQuantizedVectorValues().getCentroidDP();
        final Map<String, Object> indexParameters = indexInfo.getParameters();

        // Initialize the Faiss BBQ index in native (C++) memory. This allocates:
        //   - The HNSW graph structure (adjacency lists, entry point, max level)
        //   - Off-heap storage for quantized vectors and correction factors
        // The index is identified by a memory address (pointer) returned from C++.
        final long indexMemoryAddress =
            AccessController.doPrivileged((PrivilegedAction<Long>) () -> JNIService.initFaissBBQIndex(
                indexInfo.getTotalLiveDocs(),
                knnVectorValues.dimension(),
                indexParameters,
                centroidDp,
                quantizedVecBytes,
                indexInfo.getKnnEngine()
            ));

        try {
            doBuildAndWriteIndex(indexMemoryAddress, binarizedVectorValues, knnVectorValues, indexInfo, indexParameters, quantizedVecBytes);
        } catch (final Exception e) {
            // TODO: Deallocate the native Faiss BBQ index to prevent off-heap memory leaks.
            // This should call a JNI method to free the memory at indexMemoryAddress.
            throw e;
        }
    }

    /**
     * Performs the core index building workflow after native memory has been allocated.
     *
     * <p>The process has three distinct phases:
     * <ol>
     *   <li><b>Quantized vector transfer</b>: Copies all binary quantized vectors and their
     *       correction factors from Lucene's on-disk format into the native Faiss index's
     *       off-heap memory via {@link #passQuantizedVectorsAndCorrectionFactors}</li>
     *   <li><b>HNSW graph construction</b>: Streams document IDs in batches of 1024 to the
     *       native layer, which uses them to build the HNSW graph. The quantized vectors
     *       are already in off-heap memory from phase 1, so Faiss can access them directly
     *       by ordinal offset during graph construction</li>
     *   <li><b>Index serialization</b>: Writes the HNSW graph to disk using
     *       {@code IO_FLAG_SKIP_STORAGE}, which omits the flat vector storage section and
     *       writes a "null" placeholder instead. The quantized vectors remain in Lucene's
     *       .veb file and are accessed at search time via {@code FaissBBQFlatIndex}</li>
     * </ol>
     *
     * @param indexMemoryAddress    pointer to the native Faiss BBQ index in off-heap memory
     * @param binarizedVectorValues provides access to quantized binary codes and correction factors
     * @param knnVectorValues       iterator over document IDs associated with vectors
     * @param indexInfo             build parameters including output buffer and engine configuration
     * @param indexParameters       engine-specific parameters (e.g., HNSW M, efConstruction)
     * @param quantizedVecBytes     byte length of a single quantized vector
     */
    private void doBuildAndWriteIndex(
        final long indexMemoryAddress,
        final Lucene102BinaryQuantizedVectorsReader.BinarizedVectorValues binarizedVectorValues,
        final KNNVectorValues<?> knnVectorValues,
        final BuildIndexParams indexInfo,
        final Map<String, Object> indexParameters,
        final int quantizedVecBytes
    ) throws IOException {
        // Phase 1: Transfer all quantized vectors and their correction factors to off-heap memory.
        // After this call, the native Faiss index has all the data it needs to compute distances
        // between vectors during HNSW graph construction.
        passQuantizedVectorsAndCorrectionFactors(indexMemoryAddress, binarizedVectorValues, quantizedVecBytes, indexInfo.getKnnEngine());

        // Phase 2: Stream document IDs in batches to the native layer to build the HNSW graph.
        // We batch in groups of 16 * 1024 (16KB of int data) to balance JNI call overhead against
        // memory usage. The native side uses these doc IDs to populate the id-map layer
        // (FaissIdMapIndex) that maps vector ordinals to document IDs — this mapping is
        // essential for sparse/nested cases where doc ID != vector ordinal.
        final int batchSize = 16 * 1024;
        final int[] docIds = new int[batchSize];
        int numAdded = 0;
        while (knnVectorValues.docId() != NO_MORE_DOCS) {
            int i = 0;
            while (i < batchSize && knnVectorValues.docId() != NO_MORE_DOCS) {
                docIds[i++] = knnVectorValues.docId();
                knnVectorValues.nextDoc();
            }

            // addDocsToBBQIndex triggers HNSW insertion for the batch. The native code
            // references quantized vectors by ordinal (numAdded + offset) from the data
            // that was already transferred in Phase 1.
            JNIService.addDocsToBBQIndex(indexMemoryAddress, docIds, i, numAdded, indexInfo.getKnnEngine());
            numAdded += i;
        }

        // Phase 3: Serialize the in-memory HNSW graph to disk.
        // The IO_FLAG_SKIP_STORAGE flag (set in native code) causes Faiss to write only the
        // HNSW graph structure (adjacency lists, entry point, levels) and emit a "null" section
        // name where the flat vector storage would normally go. At search time, this "null"
        // section triggers FaissIndex.load to call a Supplier<FaissIndex> that returns a
        // FaissBBQFlatIndex backed by Lucene's BinaryQuantizedVectorsReader.
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
            JNIService.writeIndex(indexInfo.getIndexOutputWithBuffer(), indexMemoryAddress, indexInfo.getKnnEngine(), indexParameters);
            return null;
        });
    }

    /**
     * Transfers all quantized vectors and their correction factors from Lucene's on-disk
     * binary quantized format into the native Faiss index's off-heap memory.
     *
     * <h3>On-disk layout per vector</h3>
     * Each vector block in the byte buffer has the following layout:
     * <pre>
     * [binaryCode (quantizedVecBytes)] [lowerInterval (4B)] [upperInterval (4B)]
     * [additionalCorrection (4B)] [quantizedComponentSum (4B)]
     * </pre>
     * Total bytes per vector = quantizedVecBytes + 16 bytes of correction factors.
     *
     * <p>Note: The on-disk format used during index build stores quantizedComponentSum as a
     * 4-byte int. However, the native SIMD search layer (see docs/5_bulk_simd_bbq_adc.md)
     * stores it as a 2-byte uint16_t to optimize memory layout for cache-friendly SIMD access.
     * This difference is handled by the native code during the transfer.
     *
     * <h3>Correction factors explained</h3>
     * <ul>
     *   <li>{@code lowerInterval}: The lower bound (ax) of the scalar quantization interval
     *       for this vector. Used as a base offset in the ADC scoring formula.</li>
     *   <li>{@code upperInterval}: The upper bound of the quantization interval. The effective
     *       scale factor is (upperInterval - lowerInterval). For data vectors, this is used
     *       directly as {@code lx}; for query vectors, it's further scaled by FOUR_BIT_SCALE.</li>
     *   <li>{@code additionalCorrection}: A residual correction term that accounts for
     *       quantization error. Added to the final score along with the query's correction
     *       and subtracted by the centroid dot-product.</li>
     *   <li>{@code quantizedComponentSum}: The sum of all quantized component values (x1).
     *       Used in the cross-term {@code ay * lx * x1} of the ADC scoring formula.</li>
     * </ul>
     *
     * <h3>Batching strategy</h3>
     * Vectors are transferred in batches capped at ~64KB to balance JNI overhead against
     * Java heap pressure. The batch size is computed as floor(65536 / oneBlockSize).
     *
     * <h3>Byte ordering</h3>
     * All float and int values are serialized in little-endian byte order to match the
     * native (x86/ARM) memory layout, avoiding byte-swapping overhead in the C++ layer.
     *
     * @param indexMemoryAddress    pointer to the native Faiss BBQ index
     * @param binarizedVectorValues provides quantized binary codes and correction factors
     * @param quantizedVecBytes     byte length of a single quantized binary code
     */
    private void passQuantizedVectorsAndCorrectionFactors(
        final long indexMemoryAddress,
        final Lucene102BinaryQuantizedVectorsReader.BinarizedVectorValues binarizedVectorValues,
        final int quantizedVecBytes,
        final KNNEngine knnEngine
    ) throws IOException {
        // Each vector block: [quantized binary code] + 4 correction factor fields (4 bytes each)
        // lowerInterval(float) + upperInterval(float) + additionalCorrection(float) + quantizedComponentSum(int)
        final int oneBlockSize = quantizedVecBytes + Integer.BYTES * 4;

        // Cap the transfer buffer at ~64KB to limit Java heap usage per JNI call.
        // Math.ceil ensures we get at least 1 vector per batch even for very large vectors.
        final int batchSize = (int) Math.ceil(1024 * 64 / oneBlockSize);
        byte[] buffer = null;
        for (int i = 0; i < binarizedVectorValues.size(); ) {
            // Determine how many vectors to include in this batch
            final int loopSize = Math.min(binarizedVectorValues.size() - i, batchSize);
            for (int j = 0, o = 0; j < loopSize; ++j) {
                // Read the 1-bit quantized binary code for vector at ordinal (i + j).
                // The binary code length is ((dimension + 63) / 64) * 64 / 8 bytes,
                // ensuring 64-bit alignment for efficient SIMD processing.
                final byte[] binaryVector = binarizedVectorValues.getQuantizedVectorValues().vectorValue(i + j);
                if (buffer == null) {
                    // Lazily allocate the buffer on first access, sized for a full batch.
                    // Layout per vector: [binaryCode | lowerInterval | upperInterval |
                    //                     additionalCorrection | quantizedComponentSum]
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }

                // Read the correction factors for this vector. These are computed during
                // quantization by OptimizedScalarQuantizer and stored alongside the binary codes.
                final OptimizedScalarQuantizer.QuantizationResult quantizationResult =
                    binarizedVectorValues.getQuantizedVectorValues().getCorrectiveTerms(i + j);

                // Copy the quantized binary code into the buffer
                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;

                // Serialize lowerInterval as 4 bytes in little-endian order.
                // lowerInterval (ax) is the base offset of the quantization interval.
                int bits = Float.floatToRawIntBits(quantizationResult.lowerInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                // Serialize upperInterval as 4 bytes in little-endian order.
                // upperInterval defines the top of the quantization range; the effective
                // scale is (upperInterval - lowerInterval).
                bits = Float.floatToRawIntBits(quantizationResult.upperInterval());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                // Serialize additionalCorrection as 4 bytes in little-endian order.
                // This residual term compensates for quantization error in the final score.
                bits = Float.floatToRawIntBits(quantizationResult.additionalCorrection());
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);

                // Serialize quantizedComponentSum as 4 bytes in little-endian order.
                // This is the sum of all quantized values for this vector, used in the
                // cross-term (ay * lx * x1) of the ADC scoring formula.
                bits = quantizationResult.quantizedComponentSum();
                buffer[o++] = (byte) (bits);
                buffer[o++] = (byte) (bits >>> 8);
                buffer[o++] = (byte) (bits >>> 16);
                buffer[o++] = (byte) (bits >>> 24);
            }

            // Transfer this batch of packed [binaryCode + correctionFactors] to the native
            // Faiss index. The C++ side unpacks the buffer and stores vectors sequentially
            // by ordinal, so they can be referenced during HNSW graph construction.
            JNIService.passBBQVectorsWithCorrectionFactors(indexMemoryAddress, buffer, loopSize, knnEngine);

            i += loopSize;
        }
    }
}
