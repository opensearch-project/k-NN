/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.ValidationException;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.indices.ModelDao;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateVectorDimension;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createKNNMethodContextFromLegacy;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateIfCircuitBreakerIsNotTriggered;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateIfKNNPluginEnabled;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateVectorDataType;
import static org.opensearch.knn.index.mapper.ModelFieldMapper.UNSET_MODEL_DIMENSION_IDENTIFIER;

/**
 * Field Mapper for KNN vector type. Implementations of this class define what needs to be stored in Lucene's fieldType.
 * This allows us to have alternative mappings for the same field type.
 */
@Log4j2
public abstract class KNNVectorFieldMapper extends ParametrizedFieldMapper {

    public static final String CONTENT_TYPE = "knn_vector";
    public static final String KNN_FIELD = "knn_field";

    private static KNNVectorFieldMapper toType(FieldMapper in) {
        return (KNNVectorFieldMapper) in;
    }

    /**
     * Builder for KNNVectorFieldMapper. This class defines the set of parameters that can be applied to the knn_vector
     * field type
     */
    public static class Builder extends ParametrizedFieldMapper.Builder {
        protected Boolean ignoreMalformed;

        protected final Parameter<Boolean> stored = Parameter.storeParam(m -> toType(m).stored, false);
        protected final Parameter<Boolean> hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
        protected final Parameter<Integer> dimension = new Parameter<>(KNNConstants.DIMENSION, false, () -> -1, (n, c, o) -> {
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
        }, m -> toType(m).fieldType().getKnnMappingConfig().getDimension().orElse(UNSET_MODEL_DIMENSION_IDENTIFIER));

        /**
         * data_type which defines the datatype of the vector values. This is an optional parameter and
         * this is right now only relevant for lucene engine. The default value is float.
         */
        protected final Parameter<VectorDataType> vectorDataType = new Parameter<>(
            VECTOR_DATA_TYPE_FIELD,
            false,
            () -> DEFAULT_VECTOR_DATA_TYPE_FIELD,
            (n, c, o) -> VectorDataType.get((String) o),
            m -> toType(m).vectorDataType
        );

        /**
         * modelId provides a way for a user to generate the underlying library indices from an already serialized
         * model template index. If this parameter is set, it will take precedence. This parameter is only relevant for
         * library indices that require training.
         */
        protected final Parameter<String> modelId = Parameter.stringParam(
            KNNConstants.MODEL_ID,
            false,
            m -> toType(m).fieldType().getKnnMappingConfig().getModelId().orElse(null),
            null
        );

        /**
         * knnMethodContext parameter allows a user to define their k-NN library index configuration. Defaults to an L2
         * hnsw default engine index without any parameters set
         */
        protected final Parameter<KNNMethodContext> knnMethodContext = new Parameter<>(
            KNN_METHOD,
            false,
            () -> null,
            (n, c, o) -> KNNMethodContext.parse(o),
            m -> toType(m).originalKNNMethodContext
        ).setSerializer(((b, n, v) -> {
            b.startObject(n);
            v.toXContent(b, ToXContent.EMPTY_PARAMS);
            b.endObject();
        }), m -> m.getMethodComponentContext().getName()).setValidator(v -> {
            if (v == null) return;

            ValidationException validationException = null;
            if (v.isTrainingRequired()) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format(Locale.ROOT, "\"%s\" requires training.", KNN_METHOD));
            }

            ValidationException methodValidation = v.validate();
            if (methodValidation != null) {
                validationException = validationException == null ? new ValidationException() : validationException;
                validationException.addValidationErrors(methodValidation.validationErrors());
            }

            if (validationException != null) {
                throw validationException;
            }
        });

        protected final Parameter<Map<String, String>> meta = Parameter.metaParam();

        protected ModelDao modelDao;
        protected Version indexCreatedVersion;
        // KNNMethodContext that allows us to properly configure a KNNVectorFieldMapper from another
        // KNNVectorFieldMapper. To support our legacy field mapping, on parsing, if index.knn=true and no method is
        // passed, we build a KNNMethodContext using the space type, ef_construction and m that are set in the index
        // settings. However, for fieldmappers for merging, we need to be able to initialize one field mapper from
        // another (see
        // https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L98).
        // The problem is that in this case, the settings are set to empty so we cannot properly resolve the KNNMethodContext.
        // (see
        // https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L130).
        // While we could override the KNNMethodContext parameter initializer to set the knnMethodContext based on the
        // constructed KNNMethodContext from the other field mapper, this can result in merge conflict/serialization
        // exceptions. See
        // (https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L322-L324).
        // So, what we do is pass in a "resolvedKNNMethodContext" that will either be null or be set via the merge builder
        // constructor. A similar approach was taken for https://github.com/opendistro-for-elasticsearch/k-NN/issues/288
        private KNNMethodContext resolvedKNNMethodContext;

        public Builder(String name, ModelDao modelDao, Version indexCreatedVersion, KNNMethodContext resolvedKNNMethodContext) {
            super(name);
            this.modelDao = modelDao;
            this.indexCreatedVersion = indexCreatedVersion;
            this.resolvedKNNMethodContext = resolvedKNNMethodContext;
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return Arrays.asList(stored, hasDocValues, dimension, vectorDataType, meta, knnMethodContext, modelId);
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

        private void validateFlatMapper() {
            if (modelId.get() != null || knnMethodContext.get() != null) {
                throw new IllegalArgumentException("Cannot set modelId or method parameters when index.knn setting is false");
            }
        }

        @Override
        public KNNVectorFieldMapper build(BuilderContext context) {
            final MultiFields multiFieldsBuilder = this.multiFieldsBuilder.build(this, context);
            final CopyTo copyToBuilder = copyTo.build();
            final Explicit<Boolean> ignoreMalformed = ignoreMalformed(context);
            final Map<String, String> metaValue = meta.getValue();

            // Index is being created from model
            String modelIdAsString = this.modelId.get();
            if (modelIdAsString != null) {
                return ModelFieldMapper.createFieldMapper(
                    buildFullName(context),
                    name,
                    metaValue,
                    vectorDataType.getValue(),
                    modelIdAsString,
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    modelDao,
                    indexCreatedVersion
                );
            }

            // If the field mapper is using the legacy context and being constructed from another field mapper,
            // the settings will be empty. See https://github.com/opendistro-for-elasticsearch/k-NN/issues/288. In this
            // case, the input resolvedKNNMethodContext will be null and the settings wont exist (so flat mapper should
            // be used). Otherwise, we need to check the setting.
            boolean isResolvedNull = resolvedKNNMethodContext == null;
            boolean isSettingPresent = KNNSettings.IS_KNN_INDEX_SETTING.exists(context.indexSettings());
            boolean isKnnSettingNotPresentOrFalse = !isSettingPresent || !KNNSettings.IS_KNN_INDEX_SETTING.get(context.indexSettings());
            if (isResolvedNull && isKnnSettingNotPresentOrFalse) {
                validateFlatMapper();
                return FlatVectorFieldMapper.createFieldMapper(
                    buildFullName(context),
                    name,
                    metaValue,
                    vectorDataType.getValue(),
                    dimension.getValue(),
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    indexCreatedVersion
                );
            }

            // See resolvedKNNMethodContext definition for explanation
            if (isResolvedNull) {
                resolvedKNNMethodContext = this.knnMethodContext.getValue();
                setDefaultSpaceType(resolvedKNNMethodContext, vectorDataType.getValue());
                validateSpaceType(resolvedKNNMethodContext, vectorDataType.getValue());
                validateDimensions(resolvedKNNMethodContext, vectorDataType.getValue());
                validateEncoder(resolvedKNNMethodContext, vectorDataType.getValue());
            }

            // If the knnMethodContext is null at this point, that means user built the index with the legacy k-NN
            // settings to specify algo params. We need to convert this here to a KNNMethodContext so that we can
            // properly configure the rest of the index
            if (resolvedKNNMethodContext == null) {
                resolvedKNNMethodContext = createKNNMethodContextFromLegacy(context, vectorDataType.getValue(), indexCreatedVersion);
            }

            validateVectorDataType(resolvedKNNMethodContext, vectorDataType.getValue());
            resolvedKNNMethodContext.getMethodComponentContext().setIndexVersion(indexCreatedVersion);
            if (resolvedKNNMethodContext.getKnnEngine() == KNNEngine.LUCENE) {
                log.debug(String.format(Locale.ROOT, "Use [LuceneFieldMapper] mapper for field [%s]", name));
                LuceneFieldMapper.CreateLuceneFieldMapperInput createLuceneFieldMapperInput = LuceneFieldMapper.CreateLuceneFieldMapperInput
                    .builder()
                    .name(name)
                    .multiFields(multiFieldsBuilder)
                    .copyTo(copyToBuilder)
                    .ignoreMalformed(ignoreMalformed)
                    .stored(stored.getValue())
                    .hasDocValues(hasDocValues.getValue())
                    .vectorDataType(vectorDataType.getValue())
                    .indexVersion(indexCreatedVersion)
                    .originalKnnMethodContext(knnMethodContext.get())
                    .build();
                return LuceneFieldMapper.createFieldMapper(
                    buildFullName(context),
                    metaValue,
                    vectorDataType.getValue(),
                    dimension.getValue(),
                    resolvedKNNMethodContext,
                    createLuceneFieldMapperInput
                );
            }

            return MethodFieldMapper.createFieldMapper(
                buildFullName(context),
                name,
                metaValue,
                vectorDataType.getValue(),
                dimension.getValue(),
                resolvedKNNMethodContext,
                knnMethodContext.get(),
                multiFieldsBuilder,
                copyToBuilder,
                ignoreMalformed,
                stored.getValue(),
                hasDocValues.getValue(),
                indexCreatedVersion
            );
        }

        private void validateEncoder(final KNNMethodContext knnMethodContext, final VectorDataType vectorDataType) {
            if (knnMethodContext == null) {
                return;
            }

            if (VectorDataType.FLOAT == vectorDataType) {
                return;
            }

            if (knnMethodContext.getMethodComponentContext() == null) {
                return;
            }

            if (knnMethodContext.getMethodComponentContext().getParameters() == null) {
                return;
            }

            if (knnMethodContext.getMethodComponentContext().getParameters().get(METHOD_ENCODER_PARAMETER) == null) {
                return;
            }

            if (knnMethodContext.getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext == false) {
                return;
            }

            MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) knnMethodContext.getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER);

            if (ENCODER_FLAT.equals(encoderMethodComponentContext.getName()) == false) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "%s data type does not support %s encoder",
                        vectorDataType.getValue(),
                        encoderMethodComponentContext.getName()
                    )
                );
            }
        }

        private void setDefaultSpaceType(final KNNMethodContext knnMethodContext, final VectorDataType vectorDataType) {
            if (knnMethodContext == null) {
                return;
            }

            if (SpaceType.UNDEFINED == knnMethodContext.getSpaceType()) {
                if (VectorDataType.BINARY == vectorDataType) {
                    knnMethodContext.setSpaceType(SpaceType.DEFAULT_BINARY);
                } else {
                    knnMethodContext.setSpaceType(SpaceType.DEFAULT);
                }
            }
        }

        private void validateSpaceType(final KNNMethodContext knnMethodContext, final VectorDataType vectorDataType) {
            if (knnMethodContext == null) {
                return;
            }

            knnMethodContext.getSpaceType().validateVectorDataType(vectorDataType);
        }

        private KNNEngine validateDimensions(final KNNMethodContext knnMethodContext, final VectorDataType dataType) {
            final KNNEngine knnEngine;
            if (knnMethodContext != null) {
                knnEngine = knnMethodContext.getKnnEngine();
            } else {
                knnEngine = KNNEngine.DEFAULT;
            }
            if (dimension.getValue() > KNNEngine.getMaxDimensionByEngine(knnEngine)) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Dimension value cannot be greater than %s for vector: %s",
                        KNNEngine.getMaxDimensionByEngine(knnEngine),
                        name
                    )
                );
            }
            if (VectorDataType.BINARY == dataType && dimension.getValue() % 8 != 0) {
                throw new IllegalArgumentException("Dimension should be multiply of 8 for binary vector data type");
            }
            return knnEngine;
        }
    }

    public static class TypeParser implements Mapper.TypeParser {

        // Use a supplier here because in {@link org.opensearch.knn.KNNPlugin#getMappers()} the ModelDao has not yet
        // been initialized
        private Supplier<ModelDao> modelDaoSupplier;

        public TypeParser(Supplier<ModelDao> modelDaoSupplier) {
            this.modelDaoSupplier = modelDaoSupplier;
        }

        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            Builder builder = new KNNVectorFieldMapper.Builder(name, modelDaoSupplier.get(), parserContext.indexVersionCreated(), null);
            builder.parse(name, parserContext, node);

            // All <a
            // href="https://github.com/opensearch-project/OpenSearch/blob/1.0.0/server/src/main/java/org/opensearch/index/mapper/DocumentMapperParser.java#L115-L161">parsing</a>
            // is done before any mappers are built. Therefore, validation should be done during parsing
            // so that it can fail early.
            if (builder.knnMethodContext.get() != null && builder.modelId.get() != null) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Method and model can not be both specified in the mapping: %s", name)
                );
            }

            // Dimension should not be null unless modelId is used
            if (builder.dimension.getValue() == UNSET_MODEL_DIMENSION_IDENTIFIER && builder.modelId.get() == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Dimension value missing for vector: %s", name));
            }

            return builder;
        }
    }

    // We store the version of the index with the mapper as different version of Opensearch has different default
    // values of KNN engine Algorithms hyperparameters.
    protected Version indexCreatedVersion;
    protected Explicit<Boolean> ignoreMalformed;
    protected boolean stored;
    protected boolean hasDocValues;
    protected VectorDataType vectorDataType;
    protected ModelDao modelDao;

    // We need to ensure that the original KNNMethodContext as parsed is stored to initialize the
    // Builder for serialization. So, we need to store it here. This is mainly to ensure that the legacy field mapper
    // can use KNNMethodContext without messing up serialization on mapper merge
    protected KNNMethodContext originalKNNMethodContext;

    public KNNVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion,
        KNNMethodContext originalKNNMethodContext
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.ignoreMalformed = ignoreMalformed;
        this.stored = stored;
        this.hasDocValues = hasDocValues;
        this.vectorDataType = mappedFieldType.getVectorDataType();
        updateEngineStats();
        this.indexCreatedVersion = indexCreatedVersion;
        this.originalKNNMethodContext = originalKNNMethodContext;
    }

    public KNNVectorFieldMapper clone() {
        return (KNNVectorFieldMapper) super.clone();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    protected void parseCreateField(ParseContext context) throws IOException {
        parseCreateField(
            context,
            fieldType().getKnnMappingConfig().getDimension().orElseThrow(() -> new IllegalArgumentException("Dimension is not set")),
            fieldType().getVectorDataType()
        );
    }

    /**
     * Function returns a list of fields to be indexed when the vector is float type.
     *
     * @param array array of floats
     * @param fieldType {@link FieldType}
     * @return {@link List} of {@link Field}
     */
    protected List<Field> getFieldsForFloatVector(final float[] array, final FieldType fieldType) {
        final List<Field> fields = new ArrayList<>();
        fields.add(new VectorField(name(), array, fieldType));
        if (this.stored) {
            fields.add(createStoredFieldForFloatVector(name(), array));
        }
        return fields;
    }

    /**
     * Function returns a list of fields to be indexed when the vector is byte type.
     *
     * @param array array of bytes
     * @param fieldType {@link FieldType}
     * @return {@link List} of {@link Field}
     */
    protected List<Field> getFieldsForByteVector(final byte[] array, final FieldType fieldType) {
        final List<Field> fields = new ArrayList<>();
        fields.add(new VectorField(name(), array, fieldType));
        if (this.stored) {
            fields.add(createStoredFieldForByteVector(name(), array));
        }
        return fields;
    }

    /**
     * Validation checks before parsing of doc begins
     */
    protected void validatePreparse() {
        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();
    }

    /**
     * Getter for vector validator after vector parsing
     *
     * @return VectorValidator
     */
    protected abstract VectorValidator getVectorValidator();

    /**
     * Getter for per dimension validator during vector parsing
     *
     * @return PerDimensionValidator
     */
    protected abstract PerDimensionValidator getPerDimensionValidator();

    /**
     * Getter for per dimension processor during vector parsing
     *
     * @return PerDimensionProcessor
     */
    protected abstract PerDimensionProcessor getPerDimensionProcessor();

    protected void parseCreateField(ParseContext context, int dimension, VectorDataType vectorDataType) throws IOException {
        validatePreparse();

        if (VectorDataType.BINARY == vectorDataType) {
            Optional<byte[]> bytesArrayOptional = getBytesFromContext(context, dimension, vectorDataType);

            if (bytesArrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = bytesArrayOptional.get();
            getVectorValidator().validateVector(array);
            context.doc().addAll(getFieldsForByteVector(array, fieldType));
        } else if (VectorDataType.BYTE == vectorDataType) {
            Optional<byte[]> bytesArrayOptional = getBytesFromContext(context, dimension, vectorDataType);

            if (bytesArrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = bytesArrayOptional.get();
            getVectorValidator().validateVector(array);
            context.doc().addAll(getFieldsForByteVector(array, fieldType));
        } else if (VectorDataType.FLOAT == vectorDataType) {
            Optional<float[]> floatsArrayOptional = getFloatsFromContext(context, dimension);

            if (floatsArrayOptional.isEmpty()) {
                return;
            }
            final float[] array = floatsArrayOptional.get();
            getVectorValidator().validateVector(array);
            context.doc().addAll(getFieldsForFloatVector(array, fieldType));
        } else {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Cannot parse context for unsupported values provided for field [%s]", VECTOR_DATA_TYPE_FIELD)
            );
        }

        context.path().remove();
    }

    // Returns an optional array of byte values where each value in the vector is parsed as a float and validated
    // if it is a finite number without any decimals and within the byte range of [-128 to 127].
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

    Optional<float[]> getFloatsFromContext(ParseContext context, int dimension) throws IOException {
        context.path().add(simpleName());

        PerDimensionValidator perDimensionValidator = getPerDimensionValidator();
        PerDimensionProcessor perDimensionProcessor = getPerDimensionProcessor();

        ArrayList<Float> vector = new ArrayList<>();
        XContentParser.Token token = context.parser().currentToken();
        float value;
        if (token == XContentParser.Token.START_ARRAY) {
            token = context.parser().nextToken();
            while (token != XContentParser.Token.END_ARRAY) {
                value = perDimensionProcessor.process(context.parser().floatValue());
                perDimensionValidator.validate(value);
                vector.add(value);
                token = context.parser().nextToken();
            }
        } else if (token == XContentParser.Token.VALUE_NUMBER) {
            value = perDimensionProcessor.process(context.parser().floatValue());
            perDimensionValidator.validate(value);
            vector.add(value);
            context.parser().nextToken();
        } else if (token == XContentParser.Token.VALUE_NULL) {
            context.path().remove();
            return Optional.empty();
        }
        validateVectorDimension(dimension, vector.size(), vectorDataType);

        float[] array = new float[vector.size()];
        int i = 0;
        for (Float f : vector) {
            array[i++] = f;
        }
        return Optional.of(array);
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new KNNVectorFieldMapper.Builder(
            simpleName(),
            modelDao,
            indexCreatedVersion,
            fieldType().getKnnMappingConfig().getKnnMethodContext().orElse(null)
        ).init(this);
    }

    @Override
    public final boolean parsesArrayValue() {
        return true;
    }

    @Override
    public KNNVectorFieldType fieldType() {
        return (KNNVectorFieldType) super.fieldType();
    }

    @Override
    protected void doXContentBody(XContentBuilder builder, boolean includeDefaults, Params params) throws IOException {
        super.doXContentBody(builder, includeDefaults, params);
        if (includeDefaults || ignoreMalformed.explicit()) {
            builder.field(Names.IGNORE_MALFORMED, ignoreMalformed.value());
        }
    }

    /**
     * Overwrite at child level in case specific stat needs to be updated
     */
    void updateEngineStats() {}

    public static class Names {
        public static final String IGNORE_MALFORMED = "ignore_malformed";
    }

    public static class Defaults {
        public static final Explicit<Boolean> IGNORE_MALFORMED = new Explicit<>(false, false);
        public static final FieldType FIELD_TYPE = new FieldType();

        static {
            FIELD_TYPE.setTokenized(false);
            FIELD_TYPE.setIndexOptions(IndexOptions.NONE);
            FIELD_TYPE.setDocValuesType(DocValuesType.BINARY);
            FIELD_TYPE.putAttribute(KNN_FIELD, "true"); // This attribute helps to determine knn field type
            FIELD_TYPE.freeze();
        }
    }
}
