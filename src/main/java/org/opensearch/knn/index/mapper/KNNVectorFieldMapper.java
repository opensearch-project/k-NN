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
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.search.DocValuesFieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.Nullable;
import org.opensearch.common.ValidationException;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KnnCircuitBreakerException;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.lookup.SearchLookup;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;
import static org.opensearch.knn.common.KNNValidationUtil.validateFloatVectorValue;
import static org.opensearch.knn.common.KNNValidationUtil.validateVectorDimension;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.clipVectorValueToFP16Range;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.deserializeStoredVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateFP16VectorValue;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateVectorDataTypeWithEngine;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateVectorDataTypeWithKnnIndexSetting;

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

    // We store the version of the index with the mapper as different version of Opensearch has different default
    // values of KNN engine Algorithms hyperparameters.
    protected Version indexCreatedVersion;

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
                    String.format("Unable to parse [dimension] from provided value [%s] for vector [%s]", o, name)
                );
            }
            if (value <= 0) {
                throw new IllegalArgumentException(String.format("Dimension value must be greater than 0 for vector: %s", name));
            }
            return value;
        }, m -> toType(m).dimension);

        /**
         * data_type which defines the datatype of the vector values. This is an optional parameter and
         * this is right now only relevant for lucene engine. The default value is float.
         */
        private final Parameter<VectorDataType> vectorDataType = new Parameter<>(
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
        protected final Parameter<String> modelId = Parameter.stringParam(KNNConstants.MODEL_ID, false, m -> toType(m).modelId, null);

        /**
         * knnMethodContext parameter allows a user to define their k-NN library index configuration. Defaults to an L2
         * hnsw default engine index without any parameters set
         */
        protected final Parameter<KNNMethodContext> knnMethodContext = new Parameter<>(
            KNN_METHOD,
            false,
            () -> null,
            (n, c, o) -> KNNMethodContext.parse(o),
            m -> toType(m).knnMethod
        ).setSerializer(((b, n, v) -> {
            b.startObject(n);
            v.toXContent(b, ToXContent.EMPTY_PARAMS);
            b.endObject();
        }), m -> m.getMethodComponentContext().getName()).setValidator(v -> {
            if (v == null) return;

            ValidationException validationException = null;
            if (v.isTrainingRequired()) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format("\"%s\" requires training.", KNN_METHOD));
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

        protected String spaceType;
        protected String m;
        protected String efConstruction;

        protected ModelDao modelDao;

        protected Version indexCreatedVersion;

        public Builder(String name, ModelDao modelDao, Version indexCreatedVersion) {
            super(name);
            this.modelDao = modelDao;
            this.indexCreatedVersion = indexCreatedVersion;
        }

        /**
         * This constructor is for legacy purposes.
         * Checkout <a href="https://github.com/opendistro-for-elasticsearch/k-NN/issues/288">ODFE PR 288</a>
         *
         * @param name field name
         * @param spaceType Spacetype of field
         * @param m m value of field
         * @param efConstruction efConstruction value of field
         */
        public Builder(String name, String spaceType, String m, String efConstruction, Version indexCreatedVersion) {
            super(name);
            this.spaceType = spaceType;
            this.m = m;
            this.efConstruction = efConstruction;
            this.indexCreatedVersion = indexCreatedVersion;
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

        @Override
        public KNNVectorFieldMapper build(BuilderContext context) {
            // Originally, a user would use index settings to set the spaceType, efConstruction and m hnsw
            // parameters. Upon further review, it makes sense to set these parameters in the mapping of a
            // particular field. However, because users migrating from older versions will still use the index
            // settings to set these parameters, we will need to provide backwards compatibilty. In order to
            // handle this, we first check if the mapping is set, and, if so use it. If not, we check if the model is
            // set. If not, we fall back to the parameters set in the index settings. This means that if a user sets
            // the mappings, setting the index settings will have no impact.

            final KNNMethodContext knnMethodContext = this.knnMethodContext.getValue();
            validateMaxDimensions(knnMethodContext);
            final MultiFields multiFieldsBuilder = this.multiFieldsBuilder.build(this, context);
            final CopyTo copyToBuilder = copyTo.build();
            final Explicit<Boolean> ignoreMalformed = ignoreMalformed(context);
            final Map<String, String> metaValue = meta.getValue();

            if (knnMethodContext != null) {
                knnMethodContext.getMethodComponentContext().setIndexVersion(indexCreatedVersion);
                final KNNVectorFieldType mappedFieldType = new KNNVectorFieldType(
                    buildFullName(context),
                    metaValue,
                    dimension.getValue(),
                    knnMethodContext,
                    vectorDataType.getValue()
                );
                if (knnMethodContext.getKnnEngine() == KNNEngine.LUCENE) {
                    log.debug(String.format("Use [LuceneFieldMapper] mapper for field [%s]", name));
                    LuceneFieldMapper.CreateLuceneFieldMapperInput createLuceneFieldMapperInput =
                        LuceneFieldMapper.CreateLuceneFieldMapperInput.builder()
                            .name(name)
                            .mappedFieldType(mappedFieldType)
                            .multiFields(multiFieldsBuilder)
                            .copyTo(copyToBuilder)
                            .ignoreMalformed(ignoreMalformed)
                            .stored(stored.get())
                            .hasDocValues(hasDocValues.get())
                            .vectorDataType(vectorDataType.getValue())
                            .knnMethodContext(knnMethodContext)
                            .build();
                    return new LuceneFieldMapper(createLuceneFieldMapperInput);
                }

                // Validates and throws exception if data_type field is set in the index mapping
                // using any VectorDataType (other than float, which is default) because other
                // VectorDataTypes are only supported for lucene engine.
                validateVectorDataTypeWithEngine(vectorDataType);

                return new MethodFieldMapper(
                    name,
                    mappedFieldType,
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    knnMethodContext
                );
            }

            String modelIdAsString = this.modelId.get();
            if (modelIdAsString != null) {
                // Because model information is stored in cluster metadata, we are unable to get it here. This is
                // because to get the cluster metadata, you need access to the cluster state. Because this code is
                // sometimes used to initialize the cluster state/update cluster state, we cannot get the state here
                // safely. So, we are unable to validate the model. The model gets validated during ingestion.

                return new ModelFieldMapper(
                    name,
                    new KNNVectorFieldType(buildFullName(context), metaValue, -1, knnMethodContext, modelIdAsString),
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    modelDao,
                    modelIdAsString,
                    indexCreatedVersion
                );
            }

            // Build legacy
            if (this.spaceType == null) {
                this.spaceType = LegacyFieldMapper.getSpaceType(context.indexSettings());
            }

            if (this.m == null) {
                this.m = LegacyFieldMapper.getM(context.indexSettings());
            }

            if (this.efConstruction == null) {
                this.efConstruction = LegacyFieldMapper.getEfConstruction(context.indexSettings(), indexCreatedVersion);
            }

            // Validates and throws exception if index.knn is set to true in the index settings
            // using any VectorDataType (other than float, which is default) because we are using NMSLIB engine for LegacyFieldMapper
            // and it only supports float VectorDataType
            validateVectorDataTypeWithKnnIndexSetting(context.indexSettings().getAsBoolean(KNN_INDEX, false), vectorDataType);

            return new LegacyFieldMapper(
                name,
                new KNNVectorFieldType(
                    buildFullName(context),
                    metaValue,
                    dimension.getValue(),
                    vectorDataType.getValue(),
                    SpaceType.getSpace(spaceType)
                ),
                multiFieldsBuilder,
                copyToBuilder,
                ignoreMalformed,
                stored.get(),
                hasDocValues.get(),
                spaceType,
                m,
                efConstruction,
                indexCreatedVersion
            );
        }

        private KNNEngine validateMaxDimensions(final KNNMethodContext knnMethodContext) {
            final KNNEngine knnEngine;
            if (knnMethodContext != null) {
                knnEngine = knnMethodContext.getKnnEngine();
            } else {
                knnEngine = KNNEngine.DEFAULT;
            }
            if (dimension.getValue() > KNNEngine.getMaxDimensionByEngine(knnEngine)) {
                throw new IllegalArgumentException(
                    String.format(
                        "Dimension value cannot be greater than %s for vector: %s",
                        KNNEngine.getMaxDimensionByEngine(knnEngine),
                        name
                    )
                );
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
            Builder builder = new KNNVectorFieldMapper.Builder(name, modelDaoSupplier.get(), parserContext.indexVersionCreated());
            builder.parse(name, parserContext, node);

            // All <a
            // href="https://github.com/opensearch-project/OpenSearch/blob/1.0.0/server/src/main/java/org/opensearch/index/mapper/DocumentMapperParser.java#L115-L161">parsing</a>
            // is done before any mappers are built. Therefore, validation should be done during parsing
            // so that it can fail early.
            if (builder.knnMethodContext.get() != null && builder.modelId.get() != null) {
                throw new IllegalArgumentException(String.format("Method and model can not be both specified in the mapping: %s", name));
            }

            // Dimension should not be null unless modelId is used
            if (builder.dimension.getValue() == -1 && builder.modelId.get() == null) {
                throw new IllegalArgumentException(String.format("Dimension value missing for vector: %s", name));
            }

            return builder;
        }
    }

    @Getter
    public static class KNNVectorFieldType extends MappedFieldType {
        int dimension;
        String modelId;
        KNNMethodContext knnMethodContext;
        VectorDataType vectorDataType;
        SpaceType spaceType;

        public KNNVectorFieldType(
            String name,
            Map<String, String> meta,
            int dimension,
            VectorDataType vectorDataType,
            SpaceType spaceType
        ) {
            this(name, meta, dimension, null, null, vectorDataType, spaceType);
        }

        public KNNVectorFieldType(String name, Map<String, String> meta, int dimension, KNNMethodContext knnMethodContext) {
            this(name, meta, dimension, knnMethodContext, null, DEFAULT_VECTOR_DATA_TYPE_FIELD, knnMethodContext.getSpaceType());
        }

        public KNNVectorFieldType(String name, Map<String, String> meta, int dimension, KNNMethodContext knnMethodContext, String modelId) {
            this(name, meta, dimension, knnMethodContext, modelId, DEFAULT_VECTOR_DATA_TYPE_FIELD, null);
        }

        public KNNVectorFieldType(
            String name,
            Map<String, String> meta,
            int dimension,
            KNNMethodContext knnMethodContext,
            VectorDataType vectorDataType
        ) {
            this(name, meta, dimension, knnMethodContext, null, vectorDataType, knnMethodContext.getSpaceType());
        }

        public KNNVectorFieldType(
            String name,
            Map<String, String> meta,
            int dimension,
            @Nullable KNNMethodContext knnMethodContext,
            @Nullable String modelId,
            VectorDataType vectorDataType,
            @Nullable SpaceType spaceType
        ) {
            super(name, false, false, true, TextSearchInfo.NONE, meta);
            this.dimension = dimension;
            this.modelId = modelId;
            this.knnMethodContext = knnMethodContext;
            this.vectorDataType = vectorDataType;
            this.spaceType = spaceType;
        }

        @Override
        public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
            throw new UnsupportedOperationException("KNN Vector do not support fields search");
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        @Override
        public Query existsQuery(QueryShardContext context) {
            return new DocValuesFieldExistsQuery(name());
        }

        @Override
        public Query termQuery(Object value, QueryShardContext context) {
            throw new QueryShardException(
                context,
                String.format("KNN vector do not support exact searching, use KNN queries instead: [%s]", name())
            );
        }

        @Override
        public IndexFieldData.Builder fielddataBuilder(String fullyQualifiedIndexName, Supplier<SearchLookup> searchLookup) {
            failIfNoDocValues();
            return new KNNVectorIndexFieldData.Builder(name(), CoreValuesSourceType.BYTES, this.vectorDataType);
        }

        @Override
        public Object valueForDisplay(Object value) {
            return deserializeStoredVector((BytesRef) value, vectorDataType);
        }
    }

    protected Explicit<Boolean> ignoreMalformed;
    protected boolean stored;
    protected boolean hasDocValues;
    protected Integer dimension;
    protected VectorDataType vectorDataType;
    protected ModelDao modelDao;

    // These members map to parameters in the builder. They need to be declared in the abstract class due to the
    // "toType" function used in the builder. So, when adding a parameter, it needs to be added here, but set in a
    // subclass (if it is unique).
    protected KNNMethodContext knnMethod;
    protected String modelId;

    public KNNVectorFieldMapper(
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
        this.dimension = mappedFieldType.getDimension();
        this.vectorDataType = mappedFieldType.getVectorDataType();
        updateEngineStats();
        this.indexCreatedVersion = indexCreatedVersion;
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
            fieldType().getDimension(),
            fieldType().getSpaceType(),
            getMethodComponentContext(fieldType().getKnnMethodContext())
        );
    }

    private MethodComponentContext getMethodComponentContext(KNNMethodContext knnMethodContext) {
        if (Objects.isNull(knnMethodContext)) {
            return null;
        }
        return knnMethodContext.getMethodComponentContext();
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

    protected void parseCreateField(ParseContext context, int dimension, SpaceType spaceType, MethodComponentContext methodComponentContext)
        throws IOException {

        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();

        if (VectorDataType.BYTE == vectorDataType) {
            Optional<byte[]> bytesArrayOptional = getBytesFromContext(context, dimension);

            if (bytesArrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = bytesArrayOptional.get();
            spaceType.validateVector(array);
            context.doc().addAll(getFieldsForByteVector(array, fieldType));
        } else if (VectorDataType.FLOAT == vectorDataType) {
            Optional<float[]> floatsArrayOptional = getFloatsFromContext(context, dimension, methodComponentContext);

            if (floatsArrayOptional.isEmpty()) {
                return;
            }
            final float[] array = floatsArrayOptional.get();
            spaceType.validateVector(array);
            context.doc().addAll(getFieldsForFloatVector(array, fieldType));
        } else {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Cannot parse context for unsupported values provided for field [%s]", VECTOR_DATA_TYPE_FIELD)
            );
        }

        context.path().remove();
    }

    // Verify mapping and return true if it is a "faiss" Index using "sq" encoder of type "fp16"
    protected boolean isFaissSQfp16(MethodComponentContext methodComponentContext) {
        if (Objects.isNull(methodComponentContext)) {
            return false;
        }

        if (methodComponentContext.getParameters().size() == 0) {
            return false;
        }

        Map<String, Object> methodComponentParams = methodComponentContext.getParameters();

        // The method component parameters should have an encoder
        if (!methodComponentParams.containsKey(METHOD_ENCODER_PARAMETER)) {
            return false;
        }

        // Validate if the object is of type MethodComponentContext before casting it later
        if (!(methodComponentParams.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
            return false;
        }

        MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) methodComponentParams.get(METHOD_ENCODER_PARAMETER);

        // returns true if encoder name is "sq" and type is "fp16"
        return ENCODER_SQ.equals(encoderMethodComponentContext.getName())
            && FAISS_SQ_ENCODER_FP16.equals(
                encoderMethodComponentContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16)
            );

    }

    // Verify mapping and return the value of "clip" parameter(default false) for a "faiss" Index
    // using "sq" encoder of type "fp16".
    protected boolean isFaissSQClipToFP16RangeEnabled(MethodComponentContext methodComponentContext) {
        if (Objects.nonNull(methodComponentContext)) {
            return (boolean) methodComponentContext.getParameters().getOrDefault(FAISS_SQ_CLIP, false);
        }
        return false;
    }

    void validateIfCircuitBreakerIsNotTriggered() {
        if (KNNSettings.isCircuitBreakerTriggered()) {
            throw new KnnCircuitBreakerException(
                "Parsing the created knn vector fields prior to indexing has failed as the circuit breaker triggered.  This indicates that the cluster is low on memory resources and cannot index more documents at the moment. Check _plugins/_knn/stats for the circuit breaker status."
            );
        }
    }

    void validateIfKNNPluginEnabled() {
        if (!KNNSettings.isKNNPluginEnabled()) {
            throw new IllegalStateException("KNN plugin is disabled. To enable update knn.plugin.enabled setting to true");
        }
    }

    // Returns an optional array of byte values where each value in the vector is parsed as a float and validated
    // if it is a finite number without any decimals and within the byte range of [-128 to 127].
    Optional<byte[]> getBytesFromContext(ParseContext context, int dimension) throws IOException {
        context.path().add(simpleName());

        ArrayList<Byte> vector = new ArrayList<>();
        XContentParser.Token token = context.parser().currentToken();
        float value;

        if (token == XContentParser.Token.START_ARRAY) {
            token = context.parser().nextToken();
            while (token != XContentParser.Token.END_ARRAY) {
                value = context.parser().floatValue();
                validateByteVectorValue(value);
                vector.add((byte) value);
                token = context.parser().nextToken();
            }
        } else if (token == XContentParser.Token.VALUE_NUMBER) {
            value = context.parser().floatValue();
            validateByteVectorValue(value);
            vector.add((byte) value);
            context.parser().nextToken();
        } else if (token == XContentParser.Token.VALUE_NULL) {
            context.path().remove();
            return Optional.empty();
        }
        validateVectorDimension(dimension, vector.size());
        byte[] array = new byte[vector.size()];
        int i = 0;
        for (Byte f : vector) {
            array[i++] = f;
        }
        return Optional.of(array);
    }

    Optional<float[]> getFloatsFromContext(ParseContext context, int dimension, MethodComponentContext methodComponentContext)
        throws IOException {
        context.path().add(simpleName());

        // Returns an optional array of float values where each value in the vector is parsed as a float and validated
        // if it is a finite number and within the fp16 range of [-65504 to 65504] by default if Faiss encoder is SQ and type is 'fp16'.
        // If the encoder parameter, "clip" is set to True, if the vector value is outside the FP16 range then it will be
        // clipped to FP16 range.
        boolean isFaissSQfp16Flag = isFaissSQfp16(methodComponentContext);
        boolean clipVectorValueToFP16RangeFlag = false;
        if (isFaissSQfp16Flag) {
            clipVectorValueToFP16RangeFlag = isFaissSQClipToFP16RangeEnabled(
                (MethodComponentContext) methodComponentContext.getParameters().get(METHOD_ENCODER_PARAMETER)
            );
        }

        ArrayList<Float> vector = new ArrayList<>();
        XContentParser.Token token = context.parser().currentToken();
        float value;
        if (token == XContentParser.Token.START_ARRAY) {
            token = context.parser().nextToken();
            while (token != XContentParser.Token.END_ARRAY) {
                value = context.parser().floatValue();
                if (isFaissSQfp16Flag) {
                    if (clipVectorValueToFP16RangeFlag) {
                        value = clipVectorValueToFP16Range(value);
                    } else {
                        validateFP16VectorValue(value);
                    }
                } else {
                    validateFloatVectorValue(value);
                }

                vector.add(value);
                token = context.parser().nextToken();
            }
        } else if (token == XContentParser.Token.VALUE_NUMBER) {
            value = context.parser().floatValue();
            if (isFaissSQfp16Flag) {
                if (clipVectorValueToFP16RangeFlag) {
                    value = clipVectorValueToFP16Range(value);
                } else {
                    validateFP16VectorValue(value);
                }
            } else {
                validateFloatVectorValue(value);
            }
            vector.add(value);
            context.parser().nextToken();
        } else if (token == XContentParser.Token.VALUE_NULL) {
            context.path().remove();
            return Optional.empty();
        }
        validateVectorDimension(dimension, vector.size());

        float[] array = new float[vector.size()];
        int i = 0;
        for (Float f : vector) {
            array[i++] = f;
        }
        return Optional.of(array);
    }

    @Override
    public ParametrizedFieldMapper.Builder getMergeBuilder() {
        return new KNNVectorFieldMapper.Builder(simpleName(), modelDao, indexCreatedVersion).init(this);
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
