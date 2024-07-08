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

package org.opensearch.knn.quantization.models.quantizationParams;

import org.opensearch.knn.quantization.enums.QuantizationType;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

public class SQParams extends QuantizationParams {
    private SQTypes sqType;

    public SQParams(SQTypes sqType) {
        super(QuantizationType.VALUE_QUANTIZATION);
        this.sqType = sqType;
    }
    public SQTypes getSqType() {
        return sqType;
    }
}
