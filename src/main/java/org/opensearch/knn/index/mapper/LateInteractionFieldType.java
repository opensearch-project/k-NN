/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.search.lookup.SearchLookup;

import java.util.Collections;
import java.util.Locale;
import java.util.Map;

/**
 * {@link MappedFieldType} for the late-interaction (multi-vector) field.
 *
 * <p>A late-interaction field stores a variable number of equal-dimension vectors per document
 * (e.g. ColBERT / ColPali token embeddings) as Lucene {@code BinaryDocValues} via
 * {@link org.apache.lucene.document.LateInteractionField}. Unlike {@link KNNVectorFieldType}, it
 * does <b>not</b> build an ANN structure over the vectors: there is no {@code method}, {@code engine},
 * or graph. The field is scored in a second (rescore) phase using MaxSim similarity against a query
 * multi-vector.
 *
 * <p>The field is not directly searchable via term/exact queries; it is consumed by the
 * late-interaction rescore path.
 */
@Getter
public class LateInteractionFieldType extends MappedFieldType {

    /** Dimension of each individual (per-token) vector. The number of vectors per doc is variable. */
    private final int dimension;

    /** Space type used for the per-vector similarity inside MaxSim (defaults to inner product). */
    private final SpaceType spaceType;

    public LateInteractionFieldType(String name, Map<String, String> metadata, int dimension, SpaceType spaceType) {
        // indexed=false, docValues=true (stored as BinaryDocValues), stored=false, no term search.
        super(name, false, false, true, TextSearchInfo.NONE, metadata);
        this.dimension = dimension;
        this.spaceType = spaceType;
    }

    @Override
    public String typeName() {
        return LateInteractionFieldMapper.CONTENT_TYPE;
    }

    @Override
    public Query existsQuery(QueryShardContext context) {
        return new FieldExistsQuery(name());
    }

    @Override
    public Query termQuery(Object value, QueryShardContext context) {
        throw new QueryShardException(
            context,
            String.format(
                Locale.ROOT,
                "Late interaction field [%s] does not support exact/term search; use it as a rescore target instead.",
                name()
            )
        );
    }

    @Override
    public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
        // Multi-vectors are stored as binary doc values, not in a fetchable _source-friendly form here.
        return sourceLookup -> Collections.emptyList();
    }
}
