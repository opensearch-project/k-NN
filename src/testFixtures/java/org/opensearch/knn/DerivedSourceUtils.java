/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.Builder;
import lombok.Data;
import lombok.SneakyThrows;
import lombok.experimental.SuperBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.indices.replication.common.ReplicationType;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import static org.opensearch.knn.KNNRestTestCase.PROPERTIES_FIELD;
import static org.opensearch.knn.TestUtils.BWC_VERSION;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD;

public class DerivedSourceUtils {
    public static final int TEST_DIMENSION = 16;
    protected static final int DOCS = 500;

    public static final float DEFAULT_NULL_PROB = 0.03f;

    protected static Settings DERIVED_ENABLED_SETTINGS;

    public static Settings DERIVED_ENABLED_WITH_SEGREP_SETTINGS;

    protected static Settings DERIVED_DISABLED_SETTINGS;

    static {
        KNNRestTestCase knnRestTestCase = new KNNRestTestCase();
        Settings.Builder derived_enabled_settings_builder = Settings.builder()
            .put(
                "number_of_shards",
                System.getProperty(BWC_VERSION, null) == null ? Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) : 1
            )
            .put(
                "number_of_replicas",
                Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) > 1 && System.getProperty(BWC_VERSION, null) == null
                    ? 1
                    : 0
            )
            .put("index.knn", true)
            .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, true);

        Settings.Builder derived_enabled_with_segrep_settings_builder = Settings.builder()
            .put(
                "number_of_shards",
                System.getProperty(BWC_VERSION, null) == null ? Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) : 1
            )
            .put(
                "number_of_replicas",
                Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) > 1 && System.getProperty(BWC_VERSION, null) == null
                    ? 1
                    : 0
            )
            .put("index.replication.type", ReplicationType.SEGMENT.toString())
            .put("index.knn", true)
            .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, true);

        Settings.Builder derived_disabled_settings_builder = Settings.builder()
            .put(
                "number_of_shards",
                System.getProperty(BWC_VERSION, null) == null ? Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) : 1
            )
            .put(
                "number_of_replicas",
                Integer.parseInt(System.getProperty("cluster.number_of_nodes", "1")) > 1 && System.getProperty(BWC_VERSION, null) == null
                    ? 1
                    : 0
            )
            .put("index.knn", true)
            .put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, false);

        final String remoteBuild = System.getProperty("test.remoteBuild", null);
        if (knnRestTestCase.isRemoteIndexBuildSupported(knnRestTestCase.getBWCVersion()) && remoteBuild != null) {
            derived_enabled_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD, true);
            derived_enabled_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "1kb");

            derived_enabled_with_segrep_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD, true);
            derived_enabled_with_segrep_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "1kb");

            derived_disabled_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD, true);
            derived_disabled_settings_builder.put(KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD, "1kb");
        }

        DERIVED_ENABLED_SETTINGS = derived_enabled_settings_builder.build();
        DERIVED_ENABLED_WITH_SEGREP_SETTINGS = derived_enabled_with_segrep_settings_builder.build();
        DERIVED_DISABLED_SETTINGS = derived_disabled_settings_builder.build();
    }

    private static final Logger log = LogManager.getLogger(DerivedSourceUtils.class);

    @SuperBuilder
    @Data
    public static class IndexConfigContext {
        public String indexName;
        public List<FieldContext> fields;
        @Builder.Default
        public Random random = null;
        @Builder.Default
        public boolean derivedEnabled = false;
        @Builder.Default
        public int docCount = DOCS;
        @Builder.Default
        public Settings settings = null;

        public void init() {
            assert random != null;
            for (FieldContext context : fields) {
                context.init(random);
            }
        }

        public Settings getSettings() {
            if (settings != null) {
                return settings;
            }
            return derivedEnabled ? DERIVED_ENABLED_SETTINGS : DERIVED_DISABLED_SETTINGS;
        }

        @SneakyThrows
        public String getMapping() {
            XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(PROPERTIES_FIELD);
            for (FieldContext context : fields) {
                context.buildMapping(builder);
            }
            builder.endObject().endObject();
            return builder.toString();
        }

        @SneakyThrows
        public String buildDoc() {
            XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
            for (FieldContext context : fields) {
                context.buildDoc(builder);
            }
            builder.endObject();
            return builder.toString();
        }

        @SneakyThrows
        public String partialUpdateSupplier() {
            XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("doc");
            for (FieldContext context : fields) {
                context.partialUpdate(builder);
            }
            builder.endObject().endObject();
            return builder.toString();
        }

        @SneakyThrows
        public String updateByQuerySupplier(String docId) {
            XContentBuilder builder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("query")
                .startObject("term")
                .field("_id", docId)
                .endObject()
                .endObject()
                .startObject("script");

            StringBuilder sourceScript = new StringBuilder();
            for (FieldContext context : fields) {
                sourceScript.append(context.updateSourceString());
            }

            return builder.field("source", sourceScript.toString()).field("lang", "painless").endObject().endObject().toString();
        }

        @SneakyThrows
        public List<String> collectFieldNames() {
            return fields.stream().map(f -> f.fieldPath).toList();
        }
    }

    @SuperBuilder
    public static abstract class FieldContext {
        public String fieldPath;
        @Builder.Default
        public Random random = null;
        @Builder.Default
        public float skipProb = 0.1f;
        @Builder.Default
        public float nullProb = DEFAULT_NULL_PROB;
        @Builder.Default
        public boolean isUpdate = false;

        abstract XContentBuilder buildMapping(XContentBuilder builder) throws IOException;

        XContentBuilder buildDoc(XContentBuilder builder) throws IOException {
            return buildDoc(builder, skipProb, nullProb);
        }

        abstract XContentBuilder buildDoc(XContentBuilder builder, float skipProb, float nullProb) throws IOException;

        abstract XContentBuilder partialUpdate(XContentBuilder builder) throws IOException;

        public void init(Random random) {
            if (this.random == null) {
                this.random = random;
            }
        }

        protected String getFieldName() {
            String[] fields = fieldPath.split("\\.");
            return fields[fields.length - 1];
        }

        protected boolean shouldSkip(float skipProb) {
            return isUpdate == false && random.nextFloat() < skipProb;
        }

        protected boolean shouldNull(float nullProb) {
            return random.nextFloat() < nullProb;
        }

        String updateSourceString() throws IOException {
            return "";
        }
    }

    @SuperBuilder
    public abstract static class CompositeFieldContext extends FieldContext {
        public List<FieldContext> children;

        public void init(Random random) {
            super.init(random);
            for (FieldContext child : children) {
                child.init(random);
            }
        }
    }

    @SuperBuilder
    public static class ObjectFieldContext extends CompositeFieldContext {
        @Override
        XContentBuilder buildMapping(XContentBuilder builder) throws IOException {
            builder.startObject(getFieldName());
            builder.startObject("properties");
            for (FieldContext child : children) {
                child.buildMapping(builder);
            }
            builder.endObject();
            builder.endObject();
            return builder;
        }

        @Override
        XContentBuilder buildDoc(XContentBuilder builder, float skipProb, float nullProb) throws IOException {
            builder.startObject(getFieldName());
            for (FieldContext child : children) {
                child.buildDoc(builder, skipProb, nullProb);
            }
            builder.endObject();
            return builder;
        }

        @Override
        public XContentBuilder partialUpdate(XContentBuilder builder) throws IOException {
            builder.startObject(getFieldName());
            for (FieldContext child : children) {
                child.partialUpdate(builder);
            }
            builder.endObject();
            return builder;
        }

        public String updateSourceString() throws IOException {
            StringBuilder source = new StringBuilder();
            for (FieldContext child : children) {
                source.append(child.updateSourceString());
            }
            return source.toString();
        }
    }

    @SuperBuilder
    public static class NestedFieldContext extends CompositeFieldContext {
        @Builder.Default
        public int maxDocsPerNestedField = 10;
        @Builder.Default
        public int minDocsPerNestedField = 0;

        @Override
        XContentBuilder buildMapping(XContentBuilder builder) throws IOException {
            builder.startObject(getFieldName());
            builder.field("type", "nested");
            builder.startObject("properties");
            for (FieldContext child : children) {
                child.buildMapping(builder);
            }
            builder.endObject();
            builder.endObject();
            return builder;
        }

        @Override
        XContentBuilder buildDoc(XContentBuilder builder, float skipProb, float nullProb) throws IOException {
            if (shouldSkip(skipProb)) {
                return builder;
            }

            int docCount = minDocsPerNestedField + random.nextInt(maxDocsPerNestedField - minDocsPerNestedField);
            if (docCount == 1) {
                builder.startObject(getFieldName());
                for (FieldContext child : children) {
                    child.buildDoc(builder, skipProb, nullProb);
                }
                builder.endObject();
                return builder;
            }

            builder.startArray(getFieldName());
            for (int i = 0; i < docCount; i++) {
                builder.startObject();
                for (FieldContext child : children) {
                    child.buildDoc(builder, skipProb, nullProb);
                }
                builder.endObject();
            }
            builder.endArray();
            return builder;
        }

        @Override
        public XContentBuilder partialUpdate(XContentBuilder builder) throws IOException {
            return builder;
        }

    }

    @SuperBuilder
    public abstract static class LeafFieldContext extends FieldContext {
        public Supplier<Object> valueSupplier;

        @Override
        XContentBuilder buildDoc(XContentBuilder builder, float skipProb, float nullProb) throws IOException {
            if (shouldSkip(skipProb)) {
                return builder;
            }
            Object value = shouldNull(nullProb) ? null : valueSupplier.get();
            return builder.field(getFieldName(), value);
        }

        public String updateSourceString() {
            if (isUpdate) {
                Object vectorValue = valueSupplier.get();
                if (vectorValue instanceof float[]) {
                    vectorValue = Arrays.toString((float[]) vectorValue);
                }
                if (vectorValue instanceof int[]) {
                    vectorValue = Arrays.toString((int[]) vectorValue);
                }

                return "ctx._source." + fieldPath + " = " + vectorValue + "; ";
            }
            return "";
        }

        @Override
        XContentBuilder partialUpdate(XContentBuilder builder) throws IOException {
            if (isUpdate) {
                return builder.field(getFieldName(), valueSupplier.get());
            }
            return builder;
        }
    }

    @SuperBuilder
    public static class KNNVectorFieldTypeContext extends LeafFieldContext {
        @Builder.Default
        public int dimension = TEST_DIMENSION;
        @Builder.Default
        public VectorDataType vectorDataType = VectorDataType.FLOAT;

        public void init(Random random) {
            super.init(random);
            if (valueSupplier == null) {
                this.valueSupplier = randomVectorSupplier(this.random, dimension, vectorDataType);
            }
        }

        @Override
        public XContentBuilder buildMapping(XContentBuilder builder) throws IOException {
            builder.startObject(getFieldName());
            builder.field("type", "knn_vector");
            builder.field("dimension", dimension);
            builder.field("data_type", vectorDataType.getValue());
            builder.endObject();
            return builder;
        }
    }

    @SuperBuilder
    public static class TextFieldType extends LeafFieldContext {
        @Builder.Default
        public boolean isDynamic = false;

        public void init(Random random) {
            super.init(random);
            if (valueSupplier == null) {
                this.valueSupplier = () -> "test";
            }
        }

        @Override
        public XContentBuilder buildMapping(XContentBuilder builder) throws IOException {
            if (isDynamic) {
                return builder;
            }

            builder.startObject(getFieldName());
            builder.field("type", "text");
            builder.endObject();
            return builder;
        }
    }

    @SuperBuilder
    public static class IntFieldType extends LeafFieldContext {
        @Builder.Default
        public boolean isDynamic = false;

        public void init(Random random) {
            super.init(random);
            if (valueSupplier == null) {
                this.valueSupplier = () -> 1;
            }
        }

        @Override
        public XContentBuilder buildMapping(XContentBuilder builder) throws IOException {
            if (isDynamic) {
                return builder;
            }

            builder.startObject(getFieldName());
            builder.field("type", "integer");
            builder.endObject();
            return builder;
        }
    }

    public static Supplier<Object> randomVectorSupplier(Random random, int dimension, VectorDataType vectorDataType) {
        return () -> {
            switch (vectorDataType) {
                case FLOAT:
                    float[] floatVector = new float[dimension];
                    for (int i = 0; i < dimension; i++) {
                        floatVector[i] = random.nextFloat(); // Generates values between 0.0 and 1.0
                    }
                    return floatVector;

                case BYTE:
                    byte[] byteVector = new byte[dimension];
                    random.nextBytes(byteVector); // Fills the byte array with random bytes
                    return format(byteVector, vectorDataType);

                case BINARY:
                    // For binary vectors, we need to create a byte array where each byte represents 8 dimensions
                    int numBytes = (dimension + 7) / 8; // Round up to nearest byte
                    byte[] binaryVector = new byte[numBytes];
                    random.nextBytes(binaryVector);

                    // If dimension is not a multiple of 8, mask off the unused bits in the last byte
                    if (dimension % 8 != 0) {
                        int unusedBits = (numBytes * 8) - dimension;
                        byte mask = (byte) (0xFF >>> unusedBits);
                        binaryVector[numBytes - 1] &= mask;
                    }
                    return format(binaryVector, vectorDataType);

                default:
                    throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
            }
        };
    }

    private static Object format(byte[] vector, VectorDataType vectorDataType) {
        BytesRef vectorBytesRef = new BytesRef(vector);
        return KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
    }

}
