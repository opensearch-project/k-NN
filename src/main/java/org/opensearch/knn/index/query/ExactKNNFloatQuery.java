/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.VectorDataType;

import java.util.Arrays;
import java.util.Objects;

@Getter
public class ExactKNNFloatQuery extends ExactKNNQuery {
    private final float[] queryVector;

    public ExactKNNFloatQuery(
        String field,
        String spaceType,
        String indexName,
        VectorDataType vectorDataType,
        BitSetProducer parentFilter,
        float[] queryVector
    ) {
        super(field, spaceType, indexName, vectorDataType, parentFilter);
        this.queryVector = queryVector;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), Arrays.hashCode(queryVector));
    }

    public boolean equalsTo(ExactKNNQuery other) {
        if (super.equalsTo(other) == false) {
            return false;
        }
        ExactKNNFloatQuery that = (ExactKNNFloatQuery) other;
        return Arrays.equals(queryVector, that.queryVector);
    }

}
