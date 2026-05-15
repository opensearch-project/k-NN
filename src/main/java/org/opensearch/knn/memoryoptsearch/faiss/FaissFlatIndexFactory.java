/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissHNSWCagraIndex;

import java.util.Arrays;

/**
 * Factory that creates the appropriate flat storage implementation based on the field's configuration,
 * and wires it into the index tree when needed.
 */
@UtilityClass
public class FaissFlatIndexFactory {

    /**
     * Creates a {@link FaissIndex} flat storage backed by Lucene's {@link FlatVectorsReader} for
     * indices where flat storage was skipped (IO_FLAG_SKIP_STORAGE). Returns {@code null} if the field
     * does not require externally-provided flat storage.
     *
     * <p>Currently supports SQ 1-bit. To add support for other flat storage types (e.g., fp32 flat),
     * add new conditions here.
     */
    static FaissIndex createFlatIndex(final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        if (FieldInfoExtractor.isSQField(fieldInfo)
            && FieldInfoExtractor.extractSQConfig(fieldInfo).getBits() == FaissSQEncoder.Bits.ONE.getValue()) {
            return new FaissScalarQuantizedFlatIndex(flatVectorsReader, fieldInfo.getName());
        }
        return null;
    }

    /**
     * Convenience wrapper over {@link #createFlatIndex} for the binary HNSW graph, which requires
     * {@link FaissBinaryIndex}. Returns {@code null} if the created flat index is not a
     * {@link FaissBinaryIndex} or if no flat storage is needed.
     */
    static FaissBinaryIndex createBinaryIndex(final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        final FaissIndex flatIndex = createFlatIndex(fieldInfo, flatVectorsReader);
        if (flatIndex instanceof FaissBinaryIndex binaryIndex) {
            return binaryIndex;
        }
        return null;
    }

    /**
     * If the loaded FAISS index has no flat storage (e.g. SQ with 1-bit skips it via IO_FLAG_SKIP_STORAGE),
     * wires in the appropriate flat index. Handles both binary HNSW (local JNI build) and float CAGRA HNSW
     * (remote build) skip_stored_vectors indices.
     */
    static void maybeSetFlatBinaryIndex(final FaissIndex faissIndex, final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        if (!(faissIndex instanceof FaissIdMapIndex idMapIndex)) {
            return;
        }
        final FaissIndex nested = idMapIndex.getNestedIndex();

        // Binary HNSW path (local JNI build with IO_FLAG_SKIP_STORAGE)
        if (nested instanceof FaissBinaryHnswIndex binaryHnswIndex && binaryHnswIndex.getStorage() == null) {
            final FaissBinaryIndex flatBinaryIndex = createBinaryIndex(fieldInfo, flatVectorsReader);
            if (flatBinaryIndex == null) {
                throw new IllegalStateException(
                    String.format(
                        "%s found for field [%s] but %s returned null — cannot wire binary flat storage.",
                        FaissEmptyIndex.class.getName(),
                        fieldInfo.getName(),
                        FaissFlatIndexFactory.class.getName()
                    )
                );
            }
            binaryHnswIndex.setStorage(flatBinaryIndex);

            // Update the space type in FaissBinaryIndex.
            // Faiss defaults to Hamming distance for binary indices, but with scalar quantization (ADC),
            // we now support other space types (i.e. L2, inner product).
            final int metricType = idMapIndex.getMetricType();
            try {
                final SpaceType spaceType = FaissMetricType.values()[metricType].spaceType;
                idMapIndex.setSpaceType(spaceType);
            } catch (ArrayIndexOutOfBoundsException e) {
                throw new IllegalArgumentException(
                    "Unsupported metric type: " + metricType + ", only support " + Arrays.asList(FaissMetricType.values())
                );
            }
            return;
        }

        // Float CAGRA HNSW path (remote GPU build with skip_stored_vectors=true / IO_FLAG_SKIP_STORAGE)
        if (nested instanceof FaissHNSWCagraIndex cagraIndex && FaissEmptyIndex.isEmptyIndex(cagraIndex.getFlatVectors())) {
            final FaissIndex flatIndex = createFlatIndex(fieldInfo, flatVectorsReader);
            if (flatIndex == null) {
                throw new IllegalStateException(
                    String.format(
                        "%s found for field [%s] but %s returned null — cannot wire flat storage for CAGRA index.",
                        FaissEmptyIndex.class.getName(),
                        fieldInfo.getName(),
                        FaissFlatIndexFactory.class.getName()
                    )
                );
            }
            cagraIndex.setFlatVectors(flatIndex);
            // No space type override needed — float CAGRA reads the correct metric type
            // from readCommonHeader() in doLoad().
        }
    }
}
