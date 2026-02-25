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

package org.opensearch.knn;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Map;

public class NestedKnnDocBuilder {
    private XContentBuilder builder;
    private boolean isNestedFieldBuildCompleted;

    public NestedKnnDocBuilder(final String fieldName) throws IOException {
        isNestedFieldBuildCompleted = false;
        builder = XContentFactory.jsonBuilder().startObject().startArray(fieldName);
    }

    public static NestedKnnDocBuilder create(final String fieldName) throws IOException {
        return new NestedKnnDocBuilder(fieldName);
    }

    public NestedKnnDocBuilder addVectors(final String fieldName, final Object[]... vectors) throws IOException {
        for (Object[] vector : vectors) {
            builder.startObject();
            builder.field(fieldName, vector);
            builder.endObject();
        }
        return this;
    }

    public NestedKnnDocBuilder addVectorWithMetadata(
        final String fieldName,
        final Object[] vectorValue,
        final String metadataFieldName,
        final Object metadataValue
    ) throws IOException {
        builder.startObject();
        builder.field(fieldName, vectorValue);
        builder.field(metadataFieldName, metadataValue);
        builder.endObject();
        return this;
    }

    public NestedKnnDocBuilder addMetadata(final Map<String, Object> metadata) throws IOException {
        builder.startObject();
        metadata.forEach((k, v) -> {
            try {
                builder.field(k, v);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        builder.endObject();
        return this;
    }

    /**
     * Use this function when you want to add top level fields in the document that contains nested fields. Once you
     * run this function you cannot add anything in the nested field.
     */
    public NestedKnnDocBuilder addTopLevelField(final String fieldName, final Object value) throws IOException {
        if (isNestedFieldBuildCompleted == false) {
            // Making sure that we close the building of nested field.
            isNestedFieldBuildCompleted = true;
            builder.endArray();
        }
        builder.field(fieldName, value);
        return this;
    }

    public String build() throws IOException {
        if (isNestedFieldBuildCompleted) {
            builder.endObject();
        } else {
            builder.endArray().endObject();
        }
        return builder.toString();
    }
}
