/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.NumericDocValues;
import org.opensearch.common.CheckedSupplier;

import java.io.IOException;

/**
 * Supplies the L2 norm for a given document. Used to denormalize vectors when reconstructing _source.
 */
@FunctionalInterface
public interface DerivedSourceNormSupplier {

    /**
     * A no-op supplier that always returns 1.0f (no denormalization).
     */
    DerivedSourceNormSupplier UNIT = (docId) -> 1.0f;

    /**
     * Get the L2 norm for the given document.
     *
     * @param docId document ID to advance to
     * @return L2 norm value. 1.0f means no denormalization needed.
     * @throws IOException if an I/O error occurs
     */
    float getNorm(int docId) throws IOException;

    /**
     * Create a DerivedSourceNormSupplier backed by NumericDocValues.
     *
     * @param supplier supplies a fresh NumericDocValues iterator on each call
     * @return DerivedSourceNormSupplier that reads norm from doc values
     */
    static DerivedSourceNormSupplier fromDocValues(CheckedSupplier<NumericDocValues, IOException> supplier) {
        return (docId) -> {
            NumericDocValues dv = supplier.get();
            dv.advance(docId);
            return Float.intBitsToFloat((int) dv.longValue());
        };
    }
}
