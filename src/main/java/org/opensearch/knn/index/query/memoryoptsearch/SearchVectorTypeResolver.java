/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.quantizationservice.QuantizationService;

/**
 * Utility class to determine the correct vector search function for memory-optimized search operations.
 * This ensures consistency between actual search (MemoryOptimizedKNNWeight) and warmup (MemoryOptimizedSearchWarmup).
 */
public class SearchVectorTypeResolver {

    /**
     * Returns the appropriate search function based on the field configuration.
     *
     * @param reader the segment reader
     * @param fieldInfo the field information
     * @param vectorDataType the vector data type from the field
     * @return a VectorSearchFunction that performs the appropriate search
     */
    public static VectorSearchFunction getSearchFunction(
        final SegmentReader reader,
        final FieldInfo fieldInfo,
        final VectorDataType vectorDataType
    ) {
        final boolean useByteSearch = shouldUseByteVectorSearch(fieldInfo, vectorDataType);

        if (useByteSearch) {
            return (fieldName, vector, knnCollector, acceptDocs) -> reader.getVectorReader()
                .search(fieldName, (byte[]) vector, knnCollector, acceptDocs);
        } else {
            return (fieldName, vector, knnCollector, acceptDocs) -> reader.getVectorReader()
                .search(fieldName, (float[]) vector, knnCollector, acceptDocs);
        }
    }

    /**
     * Determines whether to use byte[] or float[] for the search operation based on the field configuration.
     *
     * @param fieldInfo the field information
     * @param vectorDataType the vector data type from the field
     * @return true if byte[] search should be used, false if float[] search should be used
     */
    private static boolean shouldUseByteVectorSearch(final FieldInfo fieldInfo, final VectorDataType vectorDataType) {
        // Check if quantization is configured - this determines the search vector type
        final QuantizationService<?, ?> quantizationService = QuantizationService.getInstance();
        final VectorDataType transferDataType = quantizationService.getVectorDataTypeForTransfer(fieldInfo);
        if (transferDataType != null) {
            // Quantization is configured - check if it uses BINARY transfer
            return transferDataType == VectorDataType.BINARY;
        }

        // When data_type is set to byte or binary
        if (vectorDataType == VectorDataType.BINARY || vectorDataType == VectorDataType.BYTE) {
            return true;
        }

        // Default float case (includes ADC which uses float[] search)
        return false;
    }
}
