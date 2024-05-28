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

package org.opensearch.knn.training;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

/**
 * A data spec containing relevant training data for validation.
 */
@Getter
@Setter
@AllArgsConstructor
public class TrainingDataSpec {
    private int dimension;
}
