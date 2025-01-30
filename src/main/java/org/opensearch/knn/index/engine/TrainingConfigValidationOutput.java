/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

/**
 * This object provides the output of the validation checks for training model inputs.
 * The values in this object need to be dynamically set and calling code needs to handle
 * the possibility that the values have not been set.
 */
@Setter
@Getter
@Builder
@AllArgsConstructor
public class TrainingConfigValidationOutput {
    private Boolean valid;
    private Long minTrainingVectorCount;
}
