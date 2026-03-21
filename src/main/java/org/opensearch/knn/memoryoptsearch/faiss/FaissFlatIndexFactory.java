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

import java.util.Arrays;

/**
 * Factory that creates the appropriate {@link FaissBinaryIndex} flat storage implementation
 * based on the field's configuration, and wires it into the index tree when needed.
 */
@UtilityClass
public class FaissFlatIndexFactory {

    /**
     * Returns a {@link FaissBinaryIndex} to use as flat storage for the given field, or {@code null}
     * if the FAISS file's own flat storage should be used.
     */
    static FaissBinaryIndex create(final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        if (FieldInfoExtractor.isSQField(fieldInfo)
            && FieldInfoExtractor.extractSQConfig(fieldInfo).getBits() == FaissSQEncoder.Bits.ONE.getValue()) {
            return new FaissScalarQuantizedFlatIndex(flatVectorsReader, fieldInfo.getName());
        }
        return null;
    }

    /**
     * If the loaded FAISS index has no flat storage (e.g. SQ with 1-bit skips it via IO_FLAG_SKIP_STORAGE),
     * wires in the appropriate flat index via {@link #create}.
     */
    static void maybeSetFlatBinaryIndex(final FaissIndex faissIndex, final FieldInfo fieldInfo, final FlatVectorsReader flatVectorsReader) {
        if (!(faissIndex instanceof FaissIdMapIndex idMapIndex)) {
            return;
        }
        final FaissIndex nested = idMapIndex.getNestedIndex();
        if (!(nested instanceof FaissBinaryHnswIndex binaryHnswIndex) || binaryHnswIndex.getStorage() != null) {
            return;
        }

        final FaissBinaryIndex flatBinaryIndex = create(fieldInfo, flatVectorsReader);
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
        // Update the space type accordingly.
        final int metricType = idMapIndex.getMetricType();
        if (metricType == FaissMetricType.INNER_PRODUCT.ordinal()) {
            idMapIndex.setSpaceType(SpaceType.INNER_PRODUCT);
        } else if (metricType == FaissMetricType.L2.ordinal()) {
            idMapIndex.setSpaceType(SpaceType.L2);
        } else {
            throw new IllegalArgumentException(
                "Unsupported metric type: " + metricType + ", only support " + Arrays.asList(FaissMetricType.values())
            );
        }
    }
}
