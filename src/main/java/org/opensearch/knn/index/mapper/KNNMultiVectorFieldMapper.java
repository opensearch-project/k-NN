/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateVectorDimension;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.useFullFieldNameValidation;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateIfCircuitBreakerIsNotTriggered;
import static org.opensearch.knn.index.mapper.ModelFieldMapper.UNSET_MODEL_DIMENSION_IDENTIFIER;

/**
 * "mappings": {
 *   "properties": {
 *     "my_multi_knn_vector": {
 *       "type": "multi_knn_vector"
 *       "dimension": 3
 *     }
 *   }
 * }
 *
 * PUT xx_index/_doc/1
 * { "my_multi_knn_vector": [ {1,2,3}, {4,5,6}, ...]}
 */
public class KNNMultiVectorFieldMapper extends ParametrizedFieldMapper {

    public static final String CONTENT_TYPE = "multi_knn_vector";
    public static final String MULTI_KNN_FIELD = "multi_knn_field";

    private static KNNVectorFieldMapper toType(FieldMapper in) {
        return (KNNVectorFieldMapper) in;
    }
    public static class Builder extends ParametrizedFieldMapper.Builder {
        protected Boolean ignoreMalformed;

        protected final Parameter<Boolean> stored = Parameter.storeParam(m -> toType(m).stored, false);

        protected Parameter<Boolean> hasDocValues;
        protected Version indexCreatedVersion;

        protected final Parameter<Integer> dimension = new Parameter<>(
                KNNConstants.DIMENSION,
                false,
                () -> UNSET_MODEL_DIMENSION_IDENTIFIER,
                (n, c, o) -> {
                    if (o == null) {
                        throw new IllegalArgumentException("Dimension cannot be null");
                    }
                    int value;
                    try {
                        value = XContentMapValues.nodeIntegerValue(o);
                    } catch (Exception exception) {
                        throw new IllegalArgumentException(
                                String.format(Locale.ROOT, "Unable to parse [dimension] from provided value [%s] for vector [%s]", o, name)
                        );
                    }
                    if (value <= 0) {
                        throw new IllegalArgumentException(
                                String.format(Locale.ROOT, "Dimension value must be greater than 0 for vector: %s", name)
                        );
                    }
                    return value;
                },
                m -> toType(m).originalMappingParameters.getDimension()
        );

        protected final Parameter<VectorDataType> vectorDataType = new Parameter<>(
                VECTOR_DATA_TYPE_FIELD,
                false,
                () -> DEFAULT_VECTOR_DATA_TYPE_FIELD,
                (n, c, o) -> VectorDataType.get((String) o),
                m -> toType(m).originalMappingParameters.getVectorDataType()
        );

        public Builder(
           String name,
           Version indexCreatedVersion
        ) {
            super(name);
            this.indexCreatedVersion = indexCreatedVersion;

            if (indexCreatedVersion.before(Version.V_3_0_0)) {
                hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
            } else {
                hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, false);
            }
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return Arrays.asList(stored, hasDocValues, dimension, vectorDataType);
        }

        protected Explicit<Boolean> ignoreMalformed(BuilderContext context) {
            if (ignoreMalformed != null) {
                return new Explicit<>(ignoreMalformed, true);
            }
            if (context.indexSettings() != null) {
                return new Explicit<>(IGNORE_MALFORMED_SETTING.get(context.indexSettings()), false);
            }
            return KNNVectorFieldMapper.Defaults.IGNORE_MALFORMED;
        }

        @Override
        public KNNVectorFieldMapper build(BuilderContext context) {
            if (useFullFieldNameValidation(indexCreatedVersion)) {
                validateFullFieldName(context);
            }

            final MultiFields multiFieldsBuilder = this.multiFieldsBuilder.build(this, context);
            final CopyTo copyToBuilder = copyTo.build();
            final Explicit<Boolean> ignoreMalformed = ignoreMalformed(context);

            if (indexCreatedVersion.onOrAfter(Version.V_3_0_0) && hasDocValues.isConfigured() == false) {
                hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
            }
            //TODO
            return FlatVectorFieldMapper.createFieldMapper(
                    buildFullName(context),
                    name,
                    KNNMethodConfigContext.builder()
                            .vectorDataType(vectorDataType.getValue())
                            .versionCreated(indexCreatedVersion)
                            .dimension(dimension.getValue())
                            .build(),
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get()
            );
        }

        private void validateFullFieldName(final BuilderContext context) {
            final String fullFieldName = buildFullName(context);
            for (char ch : fullFieldName.toCharArray()) {
                if (Strings.INVALID_FILENAME_CHARS.contains(ch)) {
                    throw new IllegalArgumentException(
                            String.format(
                                    Locale.ROOT,
                                    "Vector field name must not include invalid characters of %s. "
                                            + "Provided field name=[%s] had a disallowed character [%c]",
                                    Strings.INVALID_FILENAME_CHARS.stream().map(c -> "'" + c + "'").collect(Collectors.toList()),
                                    fullFieldName,
                                    ch
                            )
                    );
                }
            }
        }
    }

    public static class TypeParser implements Mapper.TypeParser {
        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            Builder builder = new KNNMultiVectorFieldMapper.Builder(name, parserContext.indexVersionCreated());
            builder.parse(name, parserContext, node);
            if (builder.dimension.getValue() != UNSET_MODEL_DIMENSION_IDENTIFIER
                    && parserContext.indexVersionCreated().onOrAfter(Version.V_2_19_0)) {
                throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "Cannot specify both a modelId and dimension in the mapping: %s", name)
                );
            }

            // Check for flat configuration and validate only if index is created after 2.17
            if (isKNNDisabled(parserContext.getSettings()) && parserContext.indexVersionCreated().onOrAfter(Version.V_2_17_0)) {
                validateFromFlat(builder);
            }

            return builder;
        }

        private boolean isKNNDisabled(Settings settings) {
            boolean isSettingPresent = KNNSettings.IS_KNN_INDEX_SETTING.exists(settings);
            return !isSettingPresent || !KNNSettings.IS_KNN_INDEX_SETTING.get(settings);
        }

        private void validateFromFlat(KNNMultiVectorFieldMapper.Builder builder) {
            validateDimensionSet(builder);
        }

        private void validateDimensionSet(KNNMultiVectorFieldMapper.Builder builder) {
            if (builder.dimension.getValue() == UNSET_MODEL_DIMENSION_IDENTIFIER) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Dimension value missing for vector: %s", builder.name()));
            }
        }
    }

    protected Version indexCreatedVersion;
    protected Explicit<Boolean> ignoreMalformed;
    protected boolean stored;
    protected boolean hasDocValues;
    protected VectorDataType vectorDataType;

    public KNNMultiVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.ignoreMalformed = ignoreMalformed;
        this.stored = stored;
        this.hasDocValues = hasDocValues;
        this.vectorDataType = mappedFieldType.getVectorDataType();
        this.indexCreatedVersion = indexCreatedVersion;
    }

    @Override
    public Builder getMergeBuilder() {
        //TODO
        return null;
    }

    public KNNMultiVectorFieldMapper clone() {
        return (KNNMultiVectorFieldMapper) super.clone();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        parseCreateField(context, fieldType().getVectorDimensions(), fieldType().getVectorDataType());
    }
    protected void validatePreparse() {
        validateIfCircuitBreakerIsNotTriggered();
    }

    //TODO
    protected void parseCreateField(ParseContext context, int dimension, VectorDataType vectorDataType) throws IOException {
        validatePreparse();

        if ()

        context.path().remove();
    }

    @Override
    public void forEach(Consumer<? super Mapper> action) {
        super.forEach(action);
    }

    @Override
    public final boolean parsesArrayValue() {
        return true;
    }

    //TODO
    @Override
    public KNNMultiVectorFieldType fieldType() {
        return (KNNMultiVectorFieldType) super.fieldType();
    }

    Optional<byte[]> getBytesFromContext(ParseContext context, int dimension, VectorDataType dataType) throws IOException {
        context.path().add(simpleName());

        PerDimensionValidator perDimensionValidator = getPerDimensionValidator();
        PerDimensionProcessor perDimensionProcessor = getPerDimensionProcessor();

        ArrayList<Byte> vector = new ArrayList<>();
        XContentParser.Token token = context.parser().currentToken();

        if (token == XContentParser.Token.START_ARRAY) {
            token = context.parser().nextToken();
            while (token != XContentParser.Token.END_ARRAY) {
                float value = perDimensionProcessor.processByte(context.parser().floatValue());
                perDimensionValidator.validateByte(value);
                vector.add((byte) value);
                token = context.parser().nextToken();
            }
        } else if (token == XContentParser.Token.VALUE_NUMBER) {
            float value = perDimensionProcessor.processByte(context.parser().floatValue());
            perDimensionValidator.validateByte(value);
            vector.add((byte) value);
            context.parser().nextToken();
        } else if (token == XContentParser.Token.VALUE_NULL) {
            context.path().remove();
            return Optional.empty();
        }
        validateVectorDimension(dimension, vector.size(), dataType);
        byte[] array = new byte[vector.size()];
        int i = 0;
        for (Byte f : vector) {
            array[i++] = f;
        }
        return Optional.of(array);
    }

    @Override
    protected void doXContentBody(XContentBuilder builder, boolean includeDefaults, Params params) throws IOException {
        super.doXContentBody(builder, includeDefaults, params);
        if (includeDefaults || ignoreMalformed.explicit()) {
            builder.field(KNNVectorFieldMapper.Names.IGNORE_MALFORMED, ignoreMalformed.value());
        }
    }
    public static class Names {
        public static final String IGNORE_MALFORMED = "ignore_malformed";
    }

    public static class Defaults {
        public static final Explicit<Boolean> IGNORE_MALFORMED = new Explicit<>(false, false);
        public static final FieldType FIELD_TYPE = new FieldType();

        static {
            FIELD_TYPE.setTokenized(false);
            FIELD_TYPE.setIndexOptions(IndexOptions.NONE);
            FIELD_TYPE.putAttribute(MULTI_KNN_FIELD, "true"); // This attribute helps to determine knn field type
            FIELD_TYPE.freeze();
        }
    }
}
