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

package org.opensearch.knn.index.query.model;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.ToString;

import java.util.Optional;

@EqualsAndHashCode
@Builder
@ToString
public class HNSWAlgoQueryParameters implements AlgoQueryParameters {

    private Integer efSearch;

    public Optional<Integer> getEfSearch() {
        return Optional.ofNullable(efSearch);
    }
}
