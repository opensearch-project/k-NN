/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

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
import java.util.Objects;

/**
 * Abstract exact k-NN query that performs brute-force distance calculations
 */
@Log4j2
@Getter
public abstract class ExactKNNQuery extends Query {

    private final String field;
    private String spaceType;
    private final String indexName;
    private final VectorDataType vectorDataType;
    private BitSetProducer parentFilter;
    private boolean expandNested;
    @Setter
    @Getter
    private boolean explain;

    protected ExactKNNQuery(
        String field,
        String spaceType,
        String indexName,
        VectorDataType vectorDataType,
        BitSetProducer parentFilter,
        boolean expandNested
    ) {
        this.field = field;
        this.spaceType = spaceType;
        this.indexName = indexName;
        this.vectorDataType = vectorDataType;
        this.parentFilter = parentFilter;
        this.expandNested = expandNested;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        return new ExactKNNWeight(this, boost);
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
        return Objects.hash(field, spaceType, indexName, vectorDataType, parentFilter, expandNested);
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) && equalsTo(getClass().cast(other));
    }

    public boolean equalsTo(ExactKNNQuery other) {
        if (other == this) return true;
        return Objects.equals(field, other.field)
            && Objects.equals(spaceType, other.spaceType)
            && Objects.equals(indexName, other.indexName)
            && Objects.equals(parentFilter, other.parentFilter)
            && Objects.equals(expandNested, other.expandNested);
    }
}
