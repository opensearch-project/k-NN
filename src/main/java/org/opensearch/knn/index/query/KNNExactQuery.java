/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.join.BitSetProducer;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Exact k-NN query that performs brute-force distance calculations.
 * Supports both regular and nested field queries with expand_nested functionality.
 */
@Log4j2
@Getter
@Builder
@AllArgsConstructor
public class KNNExactQuery extends Query {

    private final String field;
    private final float[] queryVector;
    private final byte[] byteQueryVector;
    private String spaceType;
    private final String indexName;
    private final VectorDataType vectorDataType;
    private BitSetProducer parentFilter;
    private final boolean expandNested;
    @Setter
    @Getter
    private boolean explain;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        return new KNNExactWeight(this, boost);
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public String toString(String field) {
        return field;
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, Arrays.hashCode(queryVector), Arrays.hashCode(byteQueryVector), spaceType, indexName, parentFilter);
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) && equalsTo(getClass().cast(other));
    }

    public boolean equalsTo(KNNExactQuery other) {
        if (other == this) return true;
        return Objects.equals(field, other.field)
            && Arrays.equals(queryVector, other.queryVector)
            && Arrays.equals(byteQueryVector, other.byteQueryVector)
            && Objects.equals(spaceType, other.spaceType)
            && Objects.equals(indexName, other.indexName)
            && Objects.equals(parentFilter, other.parentFilter);
    }
}
