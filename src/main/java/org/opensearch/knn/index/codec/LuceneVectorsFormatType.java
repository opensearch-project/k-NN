/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

/**
 * Enum representing the types of Lucene KNN vectors formats that can be
 * resolved
 * by {@link KNN1040BasePerFieldKnnVectorsFormat}. Each codec subclass registers the
 * format
 * types it supports in a map, and the base class routes to the appropriate
 * factory
 * based on the method context.
 *
 * <p>
 * To add a new Lucene format, add an enum value here and register a factory for
 * it
 * in the relevant codec subclass(es).
 * </p>
 */
public enum LuceneVectorsFormatType {
    /**
     * Standard HNSW format with configurable max connections and beam width.
     */
    HNSW,

    /**
     * HNSW format with scalar quantization (SQ) encoding.
     */
    SCALAR_QUANTIZED,

    /**
     * HNSW format for the optimized scalar quantizer.
     */
    OPTIMIZED_SCALAR_QUANTIZER,

    /**
     * Flat vector format (e.g., BBQ flat via Lucene).
     */
    FLAT
}
