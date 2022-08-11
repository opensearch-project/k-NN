/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.opensearch.index.query.BoolQueryBuilder;

@NoArgsConstructor
@Getter
@Setter
public class KNNQueryFilter {

    BoolQueryBuilder boolQueryBuilder;
}
