/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.search.approximate.ApproximateQuery;
import org.opensearch.search.internal.SearchContext;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * Builds the oversampled top-k phase for a quantized radial query after the search request size is available.
 *
 * <p>The candidate count is {@code ceil(size * oversample_factor)} and is capped at
 * {@link RescoreContext#MAX_FIRST_PASS_RESULTS} to bound the first-pass query.</p>
 */
public final class SizeBoundedRadialSearchQuery extends ApproximateQuery {
    private final BaseQueryFactory.CreateQueryRequest request;
    private RescoreRadialSearchQuery resolvedQuery;

    public SizeBoundedRadialSearchQuery(final BaseQueryFactory.CreateQueryRequest request) {
        this.request = Objects.requireNonNull(request);
    }

    @Override
    public boolean canApproximate(final SearchContext context) {
        final int size = context.size();
        if (size <= 0) {
            return false;
        }

        final RescoreContext rescoreContext = request.getRescoreContext().orElse(RescoreContext.getDefault());
        final int firstPassK = (int) Math.min(
            RescoreContext.MAX_FIRST_PASS_RESULTS,
            Math.ceil((double) size * rescoreContext.getOversampleFactor())
        );

        final Query approximateCandidates = KNNQueryFactory.create(
            KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(request.getKnnEngine())
                .indexName(request.getIndexName())
                .fieldName(request.getFieldName())
                .vector(request.getVector())
                .originalVector(request.getOriginalVector())
                .byteVector(request.getByteVector())
                .vectorDataType(request.getVectorDataType())
                .k(firstPassK)
                .methodParameters(request.getMethodParameters())
                .filter(request.getFilter().orElse(null))
                .context(request.getContext().orElse(null))
                .expandNested(request.isExpandNested())
                .memoryOptimizedSearchEnabled(request.isMemoryOptimizedSearchEnabled())
                .build()
        );

        resolvedQuery = new RescoreRadialSearchQuery(
            approximateCandidates,
            request.getFieldName(),
            request.getVector(),
            request.getRadius(),
            request.isMemoryOptimizedSearchEnabled(),
            firstPassK,
            size
        );
        return true;
    }

    @VisibleForTesting
    RescoreRadialSearchQuery getResolvedQuery() {
        return resolvedQuery;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        return requireResolvedQuery().createWeight(searcher, scoreMode, boost);
    }

    @Override
    public Query rewrite(final IndexSearcher indexSearcher) throws IOException {
        return resolvedQuery == null ? this : resolvedQuery.rewrite(indexSearcher);
    }

    @Override
    public String toString(final String field) {
        if (resolvedQuery != null) {
            return resolvedQuery.toString(field);
        }
        return "SizeBoundedRadialSearchQuery[field=" + request.getFieldName() + ", radius=" + request.getRadius() + "]";
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        if (resolvedQuery == null) {
            visitor.visitLeaf(this);
            return;
        }
        resolvedQuery.visit(visitor);
    }

    @Override
    public boolean equals(final Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        final SizeBoundedRadialSearchQuery that = (SizeBoundedRadialSearchQuery) other;
        return request.isExpandNested() == that.request.isExpandNested()
            && request.isMemoryOptimizedSearchEnabled() == that.request.isMemoryOptimizedSearchEnabled()
            && Objects.equals(request.getKnnEngine(), that.request.getKnnEngine())
            && Objects.equals(request.getIndexName(), that.request.getIndexName())
            && Objects.equals(request.getFieldName(), that.request.getFieldName())
            && Arrays.equals(request.getVector(), that.request.getVector())
            && Arrays.equals(request.getOriginalVector(), that.request.getOriginalVector())
            && Arrays.equals(request.getByteVector(), that.request.getByteVector())
            && Objects.equals(request.getVectorDataType(), that.request.getVectorDataType())
            && Objects.equals(request.getMethodParameters(), that.request.getMethodParameters())
            && Objects.equals(request.getFilter().orElse(null), that.request.getFilter().orElse(null))
            && Objects.equals(request.getContext().orElse(null), that.request.getContext().orElse(null))
            && Objects.equals(request.getRescoreContext().orElse(null), that.request.getRescoreContext().orElse(null))
            && Objects.equals(request.getRadius(), that.request.getRadius());
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(
            request.getKnnEngine(),
            request.getIndexName(),
            request.getFieldName(),
            request.getVectorDataType(),
            request.getMethodParameters(),
            request.getFilter().orElse(null),
            request.getContext().orElse(null),
            request.getRescoreContext().orElse(null),
            request.getRadius(),
            request.isExpandNested(),
            request.isMemoryOptimizedSearchEnabled()
        );
        result = 31 * result + Arrays.hashCode(request.getVector());
        result = 31 * result + Arrays.hashCode(request.getOriginalVector());
        result = 31 * result + Arrays.hashCode(request.getByteVector());
        return result;
    }

    private RescoreRadialSearchQuery requireResolvedQuery() {
        if (resolvedQuery == null) {
            throw new IllegalStateException("SizeBoundedRadialSearchQuery must be resolved before execution");
        }
        return resolvedQuery;
    }
}
