/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.quantization.models.requests;

import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

public abstract class TrainingRequest<T> {
    private QuantizationParams params;
    private int totalNumberOfVectors;

    public TrainingRequest(QuantizationParams params, int totalNumberOfVectors) {
        this.params = params;
        this.totalNumberOfVectors = totalNumberOfVectors;
    }

    public QuantizationParams getParams() {
        return params;
    }

    public int getTotalNumberOfVectors() {
        return totalNumberOfVectors;
    }

    public abstract T getVector();

    public abstract T getVectorByDocId(int docId);
}

