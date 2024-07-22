/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.Builder;
import lombok.NonNull;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

/**
 * Helper method to create knn query in json string
 * Here, we don't use any of the internal class so that it can mimic user request more closely.
 */
@Builder
public class KNNJsonQueryBuilder {
    @NonNull
    private String fieldName;
    @NonNull
    private Object[] vector;
    private Integer k;
    private Float minScore;
    private String nestedFieldName;
    private String filterFieldName;
    private String filterValue;

    public String getQueryString() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        if (nestedFieldName != null) {
            builder.startObject("nested");
            builder.field("path", nestedFieldName);
            builder.startObject("query");
            builder.startObject("knn");
            builder.startObject(nestedFieldName + "." + fieldName);
        } else {
            builder.startObject("knn");
            builder.startObject(fieldName);
        }

        builder.field("vector", vector);
        if (k != null) {
            builder.field("k", k);
        }
        if (minScore != null) {
            builder.field("min_score", minScore);
        }

        if (filterFieldName != null && filterValue != null) {
            builder.startObject("filter");
            builder.startObject("term");
            builder.field(filterFieldName, filterValue);
            builder.endObject();
            builder.endObject();
        }

        builder.endObject().endObject().endObject().endObject();
        if (nestedFieldName != null) {
            builder.endObject().endObject();
        }

        return builder.toString();
    }
}
