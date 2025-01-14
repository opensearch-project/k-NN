/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.partialloading.search;

import lombok.Builder;
import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.partialloading.PartialLoadingContext;

@Builder
@Getter
public class PartialLoadingSearchParameters {
    private float[] floatQueryVector;
    private byte[] byteQueryVector;
    private PartialLoadingContext partialLoadingContext;
    private int k;
    private Integer efSearch;
    private BitSet filterIdsBitSet;
    private int[] parentIds;
    private SpaceType spaceType;
    private IndexInput indexInput;
}
