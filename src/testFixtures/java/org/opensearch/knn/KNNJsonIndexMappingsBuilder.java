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
 * Helper method to create knn index mapping in json string
 * Here, we don't use any of the internal class so that it can mimic user request more closely.
 */
@Builder
public class KNNJsonIndexMappingsBuilder {
    @NonNull
    private String fieldName;
    @NonNull
    private Integer dimension;
    private String nestedFieldName;
    private String vectorDataType;
    private Method method;

    public String getIndexMapping() throws IOException {
        if (nestedFieldName != null) {
            XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(nestedFieldName)
                .field("type", "nested")
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", dimension);
            addVectorDataType(xContentBuilder);
            addMethod(xContentBuilder);
            xContentBuilder.endObject().endObject().endObject().endObject().endObject();
            return xContentBuilder.toString();
        } else {
            XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("properties")
                .startObject(fieldName)
                .field("type", "knn_vector")
                .field("dimension", dimension);
            addVectorDataType(xContentBuilder);
            addMethod(xContentBuilder);
            xContentBuilder.endObject().endObject().endObject();
            return xContentBuilder.toString();
        }
    }

    private void addVectorDataType(final XContentBuilder xContentBuilder) throws IOException {
        if (vectorDataType == null) {
            return;
        }
        xContentBuilder.field("data_type", vectorDataType);
    }

    private void addMethod(final XContentBuilder xContentBuilder) throws IOException {
        if (method == null) {
            return;
        }
        method.addTo(xContentBuilder);
    }

    @Builder
    public static class Method {
        @NonNull
        private String methodName;
        @NonNull
        private String engine;
        private String spaceType;
        private Parameters parameters;

        private void addTo(final XContentBuilder xContentBuilder) throws IOException {
            xContentBuilder.startObject("method").field("name", methodName).field("engine", engine);
            addSpaceType(xContentBuilder);
            addParameters(xContentBuilder);
            xContentBuilder.endObject();
        }

        private void addSpaceType(final XContentBuilder xContentBuilder) throws IOException {
            if (spaceType == null) {
                return;
            }
            xContentBuilder.field("space_type", spaceType);
        }

        private void addParameters(final XContentBuilder xContentBuilder) throws IOException {
            if (parameters == null) {
                return;
            }
            parameters.addTo(xContentBuilder);
        }

        @Builder
        public static class Parameters {
            private Encoder encoder;

            private void addTo(final XContentBuilder xContentBuilder) throws IOException {
                xContentBuilder.startObject("parameters");
                addEncoder(xContentBuilder);
                xContentBuilder.endObject();
            }

            private void addEncoder(final XContentBuilder xContentBuilder) throws IOException {
                if (encoder == null) {
                    return;
                }
                encoder.addTo(xContentBuilder);
            }

            @Builder
            public static class Encoder {
                @NonNull
                private String name;

                private void addTo(final XContentBuilder xContentBuilder) throws IOException {
                    xContentBuilder.startObject("encoder");
                    xContentBuilder.field("name", name);
                    xContentBuilder.endObject();
                }
            }
        }
    }
}
