/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.warmup;

import org.apache.lucene.index.FieldInfo;

import java.io.IOException;

/**
 * Interface for field warm up strategy
 */
public abstract class FieldWarmUpStrategy {

    /**
     * Warm up field
     *
     * @param field field to warm up
     * @return true if field was warmed up, false otherwise
     * @throws IOException if an I/O error occurs
     */
    public abstract boolean warmUp(FieldInfo field) throws IOException;
}
