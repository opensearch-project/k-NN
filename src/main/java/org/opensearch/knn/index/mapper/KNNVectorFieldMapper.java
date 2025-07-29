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
import java.util.stream.Collectors;

import com.google.common.annotations.VisibleForTesting;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.opensearch.Version;
import org.opensearch.common.Explicit;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.DerivedKnnByteVectorField;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.EngineResolver;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.engine.SpaceTypeResolver;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.indices.ModelDao;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNValidationUtil.validateVectorDimension;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createKNNMethodContextFromLegacy;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForByteVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.createStoredFieldForFloatVector;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.useFullFieldNameValidation;
import static org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil.validateIfCircuitBreakerIsNotTriggered;
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

    // Supported compression levels for knn_vector field type
    @VisibleForTesting
    public static final String[] MAPPING_COMPRESSION_NAMES_ARRAY = new String[] {
        CompressionLevel.NOT_CONFIGURED.getName(),
        CompressionLevel.x1.getName(),
        CompressionLevel.x2.getName(),
        CompressionLevel.x4.getName(),
        CompressionLevel.x8.getName(),
        CompressionLevel.x16.getName(),
        CompressionLevel.x32.getName() };

    /**
     * Builder for KNNVectorFieldMapper. This class defines the set of parameters that can be applied to the knn_vector
     * field type
     */
    public static class Builder extends ParametrizedFieldMapper.Builder {
        protected Boolean ignoreMalformed;

        protected final Parameter<Boolean> stored = Parameter.storeParam(m -> toType(m).stored, false);
        protected Parameter<Boolean> hasDocValues;
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

        /**
         * data_type which defines the datatype of the vector values. This is an optional parameter and
         * this is right now only relevant for lucene engine. The default value is float.
         */
        protected final Parameter<VectorDataType> vectorDataType = new Parameter<>(
            VECTOR_DATA_TYPE_FIELD,
            false,
            () -> DEFAULT_VECTOR_DATA_TYPE_FIELD,
            (n, c, o) -> VectorDataType.get((String) o),
            m -> toType(m).originalMappingParameters.getVectorDataType()
        );

        /**
         * modelId provides a way for a user to generate the underlying library indices from an already serialized
         * model template index. If this parameter is set, it will take precedence. This parameter is only relevant for
         * library indices that require training.
         */
        protected final Parameter<String> modelId = Parameter.stringParam(
            KNNConstants.MODEL_ID,
            false,
            m -> toType(m).originalMappingParameters.getModelId(),
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
            m -> toType(m).originalMappingParameters.getKnnMethodContext()
        ).setSerializer(
            // Main serializer - handles null values with nullField
            (b, f, v) -> {
                if (v == null) {
                    b.nullField(f);
                } else {
                    b.startObject(f);
                    v.toXContent(b, ToXContent.EMPTY_PARAMS);
                    b.endObject();
                }
            },
            // Conflict serializer - simple string representation for error messages
            v -> v == null ? null : v.getMethodComponentContext().getName()
        );

        protected final Parameter<String> mode = Parameter.restrictedStringParam(
            KNNConstants.MODE_PARAMETER,
            false,
            m -> toType(m).originalMappingParameters.getMode(),
            Mode.NAMES_ARRAY
        ).acceptsNull();

        protected final Parameter<String> compressionLevel = Parameter.restrictedStringParam(
            KNNConstants.COMPRESSION_LEVEL_PARAMETER,
            false,
            m -> toType(m).originalMappingParameters.getCompressionLevel(),
            MAPPING_COMPRESSION_NAMES_ARRAY
        ).acceptsNull();

        // A top level space Type field.
        protected final Parameter<String> topLevelSpaceType = Parameter.stringParam(
            KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE,
            false,
            m -> toType(m).originalMappingParameters.getTopLevelSpaceType(),
            SpaceType.UNDEFINED.getValue()
        ).setValidator(SpaceType::getSpace);

        // A top level engine field.
        protected final Parameter<String> topLevelEngine = Parameter.stringParam(
            KNNConstants.TOP_LEVEL_PARAMETER_ENGINE,
            false,
            m -> toType(m).originalMappingParameters.getTopLevelEngine(),
            KNNEngine.UNDEFINED.getName()
        ).setValidator(KNNEngine::getEngine);

        protected final Parameter<Map<String, String>> meta = Parameter.metaParam();

        protected ModelDao modelDao;
        protected Version indexCreatedVersion;
        @Setter
        @Getter
        private KNNMethodConfigContext knnMethodConfigContext;
        @Setter
        @Getter
        private OriginalMappingParameters originalParameters;

        public Builder(
            String name,
            ModelDao modelDao,
            Version indexCreatedVersion,
            KNNMethodConfigContext knnMethodConfigContext,
            OriginalMappingParameters originalParameters
        ) {
            super(name);
            this.modelDao = modelDao;
            this.indexCreatedVersion = indexCreatedVersion;
            this.knnMethodConfigContext = knnMethodConfigContext;
            this.originalParameters = originalParameters;
            /*
             * For indices created on or after OpenSearch 3.0.0, docValues
             * defaults to false when not explicitly configured. This reduces storage
             * overhead and improves indexing performance for k-NN vector fields.
             * Changing the default value breaks BwC for existing indices on a cluster.
             *
             * Behavior matrix:
             * - Index < 3.0.0: Uses original default value
             * - Index >= 3.0.0, docValues not configured: Sets to false
             * - Any version, docValues explicitly configured: Respects configured value
             */
            if (indexCreatedVersion.before(Version.V_3_0_0)) {
                hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
            } else {
                hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, false);
            }
        }

        @Override
        protected List<Parameter<?>> getParameters() {
            return Arrays.asList(
                stored,
                hasDocValues,
                dimension,
                vectorDataType,
                meta,
                knnMethodContext,
                modelId,
                mode,
                compressionLevel,
                topLevelSpaceType,
                topLevelEngine
            );
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
            final Map<String, String> metaValue = meta.getValue();

            if (modelId.get() != null) {
                return ModelFieldMapper.createFieldMapper(
                    buildFullName(context),
                    name,
                    metaValue,
                    vectorDataType.getValue(),
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    modelDao,
                    indexCreatedVersion,
                    originalParameters,
                    knnMethodConfigContext
                );
            }

            // return FlatVectorFieldMapper only for indices that are created on or after 2.17.0, for others, use
            // EngineFieldMapper to maintain backwards compatibility
            if (originalParameters.getResolvedKnnMethodContext() == null && indexCreatedVersion.onOrAfter(Version.V_2_17_0)) {
                // Prior to 3.0.0, hasDocValues defaulted to false. However, FlatVectorFieldMapper requires
                // hasDocValues to be true to maintain proper functionality for vector search operations.
                // For indices created on or after 3.0.0, we automatically set hasDocValues to true if not
                // explicitly configured to ensure consistent behavior.
                if (indexCreatedVersion.onOrAfter(Version.V_3_0_0) && hasDocValues.isConfigured() == false) {
                    hasDocValues = Parameter.docValuesParam(m -> toType(m).hasDocValues, true);
                }
                return FlatVectorFieldMapper.createFieldMapper(
                    buildFullName(context),
                    name,
                    metaValue,
                    KNNMethodConfigContext.builder()
                        .vectorDataType(vectorDataType.getValue())
                        .versionCreated(indexCreatedVersion)
                        .dimension(dimension.getValue())
                        .build(),
                    multiFieldsBuilder,
                    copyToBuilder,
                    ignoreMalformed,
                    stored.get(),
                    hasDocValues.get(),
                    originalParameters
                );
            }

            return EngineFieldMapper.createFieldMapper(
                buildFullName(context),
                name,
                metaValue,
                knnMethodConfigContext,
                multiFieldsBuilder,
                copyToBuilder,
                ignoreMalformed,
                stored.getValue(),
                hasDocValues.get(),
                originalParameters
            );
        }

        /**
         * Validate whether provided full field name contain any invalid characters for physical file name.
         * At the moment, we use a field name as a part of file name while we throw an exception
         * if a physical file name contains any invalid characters when creating snapshot.
         * To prevent from this happening, we restrict vector field name and make sure generated file to have a valid name.
         *
         * @param context : Builder context to have field name info.
         */
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

        // Use a supplier here because in {@link org.opensearch.knn.KNNPlugin#getMappers()} the ModelDao has not yet
        // been initialized
        private Supplier<ModelDao> modelDaoSupplier;

        public TypeParser(Supplier<ModelDao> modelDaoSupplier) {
            this.modelDaoSupplier = modelDaoSupplier;
        }

        @Override
        public Mapper.Builder<?> parse(String name, Map<String, Object> node, ParserContext parserContext) throws MapperParsingException {
            Builder builder = new KNNVectorFieldMapper.Builder(
                name,
                modelDaoSupplier.get(),
                parserContext.indexVersionCreated(),
                null,
                null
            );
            builder.parse(name, parserContext, node);
            builder.setOriginalParameters(new OriginalMappingParameters(builder));

            // All <a
            // href="https://github.com/opensearch-project/OpenSearch/blob/1.0.0/server/src/main/java/org/opensearch/index/mapper/DocumentMapperParser.java#L115-L161">parsing</a>
            // is done before any mappers are built. Therefore, validation should be done during parsing
            // so that it can fail early.
            if (builder.knnMethodContext.get() != null && builder.modelId.get() != null) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Method and model can not be both specified in the mapping: %s", name)
                );
            }

            // Ensure user-defined dimension and model are mutually exclusive for indicies created after 2.19
            if (builder.dimension.getValue() != UNSET_MODEL_DIMENSION_IDENTIFIER
                && builder.modelId.get() != null
                && parserContext.indexVersionCreated().onOrAfter(Version.V_2_19_0)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Cannot specify both a modelId and dimension in the mapping: %s", name)
                );
            }

            // Check for flat configuration and validate only if index is created after 2.17
            if (isKNNDisabled(parserContext.getSettings()) && parserContext.indexVersionCreated().onOrAfter(Version.V_2_17_0)) {
                validateFromFlat(builder);
            } else if (builder.modelId.get() != null) {
                validateFromModel(builder);
            } else {
                // Validate that the mode and compression are not set if data type is not float, as they are not supported.
                // Also, validate that the index created version is on or after 2.17 as mode and compression are not supported for
                // the indices that are created before 2.17.
                validateModeAndCompression(builder, parserContext.indexVersionCreated());
                // If the original knnMethodContext is not null, resolve its space type and engine from the rest of the
                // configuration. This is consistent with the existing behavior for space type in 2.16 where we modify the
                // parsed value
                final SpaceType resolvedSpaceType = SpaceTypeResolver.INSTANCE.resolveSpaceType(
                    builder.originalParameters.getKnnMethodContext(),
                    builder.topLevelSpaceType.get(),
                    parserContext.getSettings(),
                    builder.vectorDataType.get()
                );

                // Set space type to the original knnMethodContext. Since the resolved one can be UNDEFINED,
                // we must wrap it and pick up the default when it is UNDEFINED.
                setSpaceType(builder.originalParameters.getKnnMethodContext(), resolvedSpaceType);
                validateSpaceType(builder);

                // Resolve method component. For the legacy case where space type can be configured at index level,
                // it first tries to use the given one then tries to get it from index setting when the space type is UNDEFINED.
                resolveKNNMethodComponents(builder, parserContext, resolvedSpaceType);
                // Validate if the KNN engine is allowed for index creation
                validateBlockedKNNEngine(builder.knnMethodContext.get(), parserContext.indexVersionCreated());
                validateFromKNNMethod(builder);
            }

            return builder;
        }

        private void validateSpaceType(KNNVectorFieldMapper.Builder builder) {
            final KNNMethodContext knnMethodContext = builder.knnMethodContext.get();
            // if context is defined
            if (knnMethodContext != null) {
                // now ensure both space types are same.
                final SpaceType knnMethodContextSpaceType = knnMethodContext.getSpaceType();
                final SpaceType topLevelSpaceType = SpaceType.getSpace(builder.topLevelSpaceType.get());
                if (topLevelSpaceType != SpaceType.UNDEFINED
                    && topLevelSpaceType != knnMethodContextSpaceType
                    && knnMethodContextSpaceType != SpaceType.UNDEFINED) {
                    throw new MapperParsingException(
                        "Space type in \"method\" and top level space type should be same or one of them should be defined"
                    );
                }
            }
        }

        private void validateModeAndCompression(KNNVectorFieldMapper.Builder builder, Version indexCreatedVersion) {
            boolean isModeOrCompressionConfigured = builder.mode.isConfigured() || builder.compressionLevel.isConfigured();
            if (isModeOrCompressionConfigured && builder.vectorDataType.getValue() != VectorDataType.FLOAT) {
                throw new MapperParsingException(
                    String.format(Locale.ROOT, "Compression and mode cannot be used for non-float32 data type for field %s", builder.name)
                );
            }

            if (isModeOrCompressionConfigured && indexCreatedVersion.before(Version.V_2_17_0)) {
                throw new MapperParsingException("Compression and mode can only be used on indices created on or after version 2.17.0");
            }
        }

        /**
         * Validates whether the provided KNN engine is allowed for index creation.
         * If the engine is NMSLIB and the OpenSearch version is 3.0.0 or later,
         * it throws an IllegalArgumentException to prevent new index creation.
         *
         * @param knnMethodContext The KNN method configuration that contains the engine type.
         * @param indexVersionCreated The OpenSearch version when the index is being created.
         * @throws IllegalArgumentException if the engine is NMSLIB and version is 3.0.0 or later.
         */
        private void validateBlockedKNNEngine(KNNMethodContext knnMethodContext, Version indexVersionCreated) {
            if (knnMethodContext == null) return;
            KNNEngine engine = knnMethodContext.getKnnEngine();
            if (engine.isRestricted(indexVersionCreated)) {
                throw new IllegalArgumentException(
                    engine.getName()
                        + " engine is deprecated in OpenSearch "
                        + " and cannot be used for new index creation in OpenSearch from  "
                        + engine.getRestrictedFromVersion()
                        + "."
                );
            }
        }

        private void validateFromFlat(KNNVectorFieldMapper.Builder builder) {
            if (builder.modelId.get() != null || builder.knnMethodContext.get() != null) {
                throw new IllegalArgumentException("Cannot set modelId or method parameters when index.knn setting is false");
            }
            validateDimensionSet(builder);
            validateCompressionAndModeNotSet(builder, builder.name(), "flat");
        }

        private void validateFromModel(KNNVectorFieldMapper.Builder builder) {
            // Dimension should not be null unless modelId is used
            if (builder.dimension.getValue() == UNSET_MODEL_DIMENSION_IDENTIFIER && builder.modelId.get() == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Dimension value missing for vector: %s", builder.name()));
            }
            // ensure model and top level spaceType is not defined
            if (builder.modelId.get() != null && SpaceType.getSpace(builder.topLevelSpaceType.get()) != SpaceType.UNDEFINED) {
                throw new IllegalArgumentException("TopLevel Space type and model can not be both specified in the " + "mapping");
            }

            validateCompressionAndModeNotSet(builder, builder.name(), "model");
        }

        private void validateFromKNNMethod(KNNVectorFieldMapper.Builder builder) {
            ValidationException validationException;
            if (builder.originalParameters.getResolvedKnnMethodContext().isTrainingRequired()) {
                validationException = new ValidationException();
                validationException.addValidationError(String.format(Locale.ROOT, "\"%s\" requires training.", KNN_METHOD));
                throw validationException;
            }

            if (builder.originalParameters.getResolvedKnnMethodContext() != null) {
                validationException = builder.originalParameters.getResolvedKnnMethodContext().validate(builder.knnMethodConfigContext);
                if (validationException != null) {
                    throw validationException;
                }
            }
            validateDimensionSet(builder);
        }

        private void validateDimensionSet(KNNVectorFieldMapper.Builder builder) {
            if (builder.dimension.getValue() == UNSET_MODEL_DIMENSION_IDENTIFIER) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "Dimension value missing for vector: %s", builder.name()));
            }
        }

        private void validateCompressionAndModeNotSet(KNNVectorFieldMapper.Builder builder, String name, String context) {
            if (builder.mode.isConfigured() || builder.compressionLevel.isConfigured()) {
                throw new MapperParsingException(
                    String.format(
                        Locale.ROOT,
                        "Compression and mode can not be specified in a %s mapping configuration for field: %s",
                        context,
                        name
                    )
                );
            }
        }

        private void resolveKNNMethodComponents(
            KNNVectorFieldMapper.Builder builder,
            ParserContext parserContext,
            SpaceType resolvedSpaceType
        ) {
            // Setup the initial configuration that is used to help resolve parameters.
            builder.setKnnMethodConfigContext(
                KNNMethodConfigContext.builder()
                    .vectorDataType(builder.originalParameters.getVectorDataType())
                    .versionCreated(parserContext.indexVersionCreated())
                    .dimension(builder.originalParameters.getDimension())
                    .mode(Mode.fromName(builder.originalParameters.getMode()))
                    .compressionLevel(CompressionLevel.fromName(builder.originalParameters.getCompressionLevel()))
                    .build()
            );

            if (useKNNMethodContextFromLegacy(builder, parserContext)) {
                // Then create KNNMethodContext to be used from the legacy index settings
                builder.originalParameters.setResolvedKnnMethodContext(
                    createKNNMethodContextFromLegacy(parserContext.getSettings(), parserContext.indexVersionCreated(), resolvedSpaceType)
                );
            }

            // Based on config context, if the user does not set the engine, resolve and set it
            KNNEngine resolvedKNNEngine = EngineResolver.INSTANCE.resolveEngine(
                builder.knnMethodConfigContext,
                builder.originalParameters.getResolvedKnnMethodContext(),
                builder.topLevelEngine.get(),
                false,
                builder.indexCreatedVersion
            );
            setEngine(builder.originalParameters.getResolvedKnnMethodContext(), resolvedKNNEngine);
            // Create a copy of the KNNMethodContext and fill in the parameters left blank by configuration context context
            ResolvedMethodContext resolvedMethodContext = resolvedKNNEngine.resolveMethod(
                builder.originalParameters.getResolvedKnnMethodContext(),
                builder.knnMethodConfigContext,
                false,
                resolvedSpaceType
            );

            // The original parameters stores both the resolveMethodContext as well as the original provided by the
            // user. Now that we have resolved, we need to update this in the original parameters.
            builder.originalParameters.setResolvedKnnMethodContext(resolvedMethodContext.getKnnMethodContext());
            builder.knnMethodConfigContext.setCompressionLevel(resolvedMethodContext.getCompressionLevel());
        }

        private boolean isKNNDisabled(Settings settings) {
            boolean isSettingPresent = KNNSettings.IS_KNN_INDEX_SETTING.exists(settings);
            return !isSettingPresent || !KNNSettings.IS_KNN_INDEX_SETTING.get(settings);
        }

        private void setSpaceType(final KNNMethodContext knnMethodContext, final SpaceType spaceType) {
            if (knnMethodContext == null) {
                return;
            }
            knnMethodContext.setSpaceType(spaceType);
        }

        private void setEngine(final KNNMethodContext knnMethodContext, KNNEngine knnEngine) {
            if (knnMethodContext == null || knnMethodContext.isEngineConfigured()) {
                return;
            }
            knnMethodContext.setKnnEngine(knnEngine);
        }
    }

    static boolean useKNNMethodContextFromLegacy(Builder builder, Mapper.TypeParser.ParserContext parserContext) {
        // If the original parameters are from legacy, and it is created on or before 2_17_2 since default is changed to
        // FAISS starting 2_18, which doesn't support accepting algo params from index settings
        return parserContext.indexVersionCreated().onOrBefore(Version.V_2_17_2) && builder.originalParameters.isLegacyMapping();
    }

    // We store the version of the index with the mapper as different version of Opensearch has different default
    // values of KNN engine Algorithms hyperparameters.
    protected Version indexCreatedVersion;
    protected Explicit<Boolean> ignoreMalformed;
    protected boolean stored;
    protected boolean hasDocValues;
    protected VectorDataType vectorDataType;
    protected ModelDao modelDao;
    protected boolean useLuceneBasedVectorField;
    protected Boolean isDerivedSourceEnabled;

    // We need to ensure that the original KNNMethodContext as parsed is stored to initialize the
    // Builder for serialization. So, we need to store it here. This is mainly to ensure that the legacy field mapper
    // can use KNNMethodContext without messing up serialization on mapper merge
    protected OriginalMappingParameters originalMappingParameters;

    public KNNVectorFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        Version indexCreatedVersion,
        OriginalMappingParameters originalMappingParameters
    ) {
        super(simpleName, mappedFieldType, multiFields, copyTo);
        this.ignoreMalformed = ignoreMalformed;
        this.stored = stored;
        this.hasDocValues = hasDocValues;
        this.vectorDataType = mappedFieldType.getVectorDataType();
        updateEngineStats();
        this.indexCreatedVersion = indexCreatedVersion;
        this.originalMappingParameters = originalMappingParameters;
        this.isDerivedSourceEnabled = null;
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
        parseCreateField(context, fieldType().getKnnMappingConfig().getDimension(), fieldType().getVectorDataType());
    }

    private Field createVectorField(float[] vectorValue, boolean isDerivedEnabled) {
        if (useLuceneBasedVectorField) {
            return new DerivedKnnFloatVectorField(name(), vectorValue, fieldType, isDerivedEnabled);
        }
        return new VectorField(name(), vectorValue, fieldType);
    }

    private Field createVectorField(byte[] vectorValue, boolean isDerivedEnabled) {
        if (useLuceneBasedVectorField) {
            return new DerivedKnnByteVectorField(name(), vectorValue, fieldType, isDerivedEnabled);
        }
        return new VectorField(name(), vectorValue, fieldType);
    }

    /**
     * Function returns a list of fields to be indexed when the vector is float type.
     *
     * @param array array of floats
     * @return {@link List} of {@link Field}
     */
    protected List<Field> getFieldsForFloatVector(final float[] array, boolean isDerivedEnabled) {
        final List<Field> fields = new ArrayList<>();
        fields.add(createVectorField(array, isDerivedEnabled));
        if (this.stored) {
            fields.add(createStoredFieldForFloatVector(name(), array));
        }
        return fields;
    }

    /**
     * Function returns a list of fields to be indexed when the vector is byte type.
     *
     * @param array array of bytes
     * @return {@link List} of {@link Field}
     */
    protected List<Field> getFieldsForByteVector(final byte[] array, boolean isDerivedEnabled) {
        final List<Field> fields = new ArrayList<>();
        fields.add(createVectorField(array, isDerivedEnabled));
        if (this.stored) {
            fields.add(createStoredFieldForByteVector(name(), array));
        }
        return fields;
    }

    /**
     * Validation checks before parsing of doc begins
     */
    protected void validatePreparse() {
        validateIfCircuitBreakerIsNotTriggered();
    }

    /**
     * Getter for vector validator after vector parsing
     *
     * @return VectorValidator
     */
    protected abstract VectorValidator getVectorValidator();

    /**
     * Getter for per dimension validator during vector parsing, and before any transformation
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

    /**
     * Retrieves the vector transformer for the KNN vector field.
     * This method provides access to the vector transformer instance that will be used
     * for processing vectors in the KNN field. The transformer is responsible for any
     * necessary vector transformations before indexing or searching.
     * This implementation delegates to the VectorTransformerFactory to obtain
     * the appropriate transformer instance. The returned transformer is typically
     * stateless and thread-safe.
     *
     * @return VectorTransformer An instance of VectorTransformer that will be used
     *         for vector transformations in this field
     *
     */
    protected VectorTransformer getVectorTransformer() {
        return VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER;
    }

    protected void parseCreateField(ParseContext context, int dimension, VectorDataType vectorDataType) throws IOException {
        validatePreparse();

        if (VectorDataType.BINARY == vectorDataType || VectorDataType.BYTE == vectorDataType) {
            Optional<byte[]> bytesArrayOptional = getBytesFromContext(context, dimension, vectorDataType);
            if (bytesArrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = bytesArrayOptional.get();
            getVectorValidator().validateVector(array);
            getVectorTransformer().transform(array);
            context.doc().addAll(getFieldsForByteVector(array, isDerivedEnabled(context)));
        } else if (VectorDataType.FLOAT == vectorDataType) {
            Optional<float[]> floatsArrayOptional = getFloatsFromContext(context, dimension);

            if (floatsArrayOptional.isEmpty()) {
                return;
            }
            final float[] array = floatsArrayOptional.get();
            getVectorValidator().validateVector(array);
            getVectorTransformer().transform(array);
            context.doc().addAll(getFieldsForFloatVector(array, isDerivedEnabled(context)));
        } else {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Cannot parse context for unsupported values provided for field [%s]", VECTOR_DATA_TYPE_FIELD)
            );
        }

        context.path().remove();
    }

    private boolean isDerivedEnabled(ParseContext parseContext) {
        if (isDerivedSourceEnabled == null) {
            isDerivedSourceEnabled = IndexUtil.isDerivedEnabledForIndex(parseContext.mapperService())
                && IndexUtil.isDerivedEnabledForField(fieldType(), parseContext.mapperService());
        }
        return isDerivedSourceEnabled;
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
        // We cannot get the dimension from the model based indices at this field because the
        // cluster state may not be available. So, we need to set it to null.
        KNNMethodConfigContext knnMethodConfigContext;
        if (fieldType().getKnnMappingConfig().getModelId().isPresent()) {
            knnMethodConfigContext = null;
        } else {
            knnMethodConfigContext = KNNMethodConfigContext.builder()
                .vectorDataType(vectorDataType)
                .versionCreated(indexCreatedVersion)
                .dimension(fieldType().getKnnMappingConfig().getDimension())
                .compressionLevel(fieldType().getKnnMappingConfig().getCompressionLevel())
                .mode(fieldType().getKnnMappingConfig().getMode())
                .build();
        }

        return new KNNVectorFieldMapper.Builder(
            simpleName(),
            modelDao,
            indexCreatedVersion,
            knnMethodConfigContext,
            originalMappingParameters
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
            FIELD_TYPE.putAttribute(KNN_FIELD, "true"); // This attribute helps to determine knn field type
            FIELD_TYPE.freeze();
        }
    }
}
