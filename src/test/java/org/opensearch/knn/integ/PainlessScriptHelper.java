/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNMethodContext;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class PainlessScriptHelper {

    /**
     * Utility to create a Index Mapping with multiple fields
     */
    public static String createMapping(List<MappingProperty> properties) throws IOException {
        Objects.requireNonNull(properties);
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder().startObject().startObject("properties");
        for (MappingProperty property : properties) {
            XContentBuilder builder = xContentBuilder.startObject(property.getName()).field("type", property.getType());
            if (property.getDimension() != null) {
                builder.field("dimension", property.getDimension());
            }

            if (property.getDocValues() != null) {
                builder.field("doc_values", property.getDocValues());
            }

            if (property.getKnnMethodContext() != null) {
                builder.startObject(KNNConstants.KNN_METHOD);
                property.getKnnMethodContext().toXContent(builder, ToXContent.EMPTY_PARAMS);
                builder.endObject();
            }

            builder.endObject();
        }
        xContentBuilder.endObject().endObject();
        return xContentBuilder.toString();
    }

    static class MappingProperty {

        private final String name;
        private final String type;
        private String dimension;

        private KNNMethodContext knnMethodContext;
        private Boolean docValues;

        MappingProperty(String name, String type) {
            this.name = name;
            this.type = type;
        }

        MappingProperty dimension(String dimension) {
            this.dimension = dimension;
            return this;
        }

        MappingProperty knnMethodContext(KNNMethodContext knnMethodContext) {
            this.knnMethodContext = knnMethodContext;
            return this;
        }

        MappingProperty docValues(boolean docValues) {
            this.docValues = docValues;
            return this;
        }

        KNNMethodContext getKnnMethodContext() {
            return knnMethodContext;
        }

        String getDimension() {
            return dimension;
        }

        String getName() {
            return name;
        }

        String getType() {
            return type;
        }

        Boolean getDocValues() {
            return docValues;
        }
    }
}
