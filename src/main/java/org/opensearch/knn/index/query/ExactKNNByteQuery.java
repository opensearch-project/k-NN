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
public class ExactKNNByteQuery extends ExactKNNQuery {
    private final byte[] byteQueryVector;

    public ExactKNNByteQuery(
        String field,
        String spaceType,
        String indexName,
        VectorDataType vectorDataType,
        BitSetProducer parentFilter,
        byte[] byteQueryVector
    ) {
        super(field, spaceType, indexName, vectorDataType, parentFilter);
        this.byteQueryVector = byteQueryVector;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), Arrays.hashCode(byteQueryVector));
    }

    public boolean equalsTo(ExactKNNQuery other) {
        if (super.equalsTo(other) == false) {
            return false;
        }
        ExactKNNByteQuery that = (ExactKNNByteQuery) other;
        return Arrays.equals(byteQueryVector, that.byteQueryVector);
    }
}
