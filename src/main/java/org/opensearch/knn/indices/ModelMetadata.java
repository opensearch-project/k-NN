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

package org.opensearch.knn.indices;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.config.CompressionConfig;
import org.opensearch.knn.index.engine.config.WorkloadModeConfig;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

import static org.opensearch.core.xcontent.DeprecationHandler.IGNORE_DEPRECATIONS;

@EqualsAndHashCode
@Log4j2
public class ModelMetadata implements Writeable, ToXContentObject {

    public static final String DELIMITER = ",";

    @Getter
    private final KNNEngine knnEngine;
    @Getter
    private final SpaceType spaceType;
    @Getter
    private final int dimension;
    private final AtomicReference<ModelState> state;
    @Getter
    private final String timestamp;
    @Getter
    private final String description;
    private final String trainingNodeAssignment;
    @Getter
    private final VectorDataType vectorDataType;
    @Getter
    private final MethodComponentContext methodComponentContext;
    @Getter
    private String error;
    @Getter
    private final WorkloadModeConfig workloadModeConfig;
    @Getter
    private final CompressionConfig compressionConfig;

    /**
     * Constructor
     *
     * @param in Stream input
     */
    public ModelMetadata(StreamInput in) throws IOException {
        this.knnEngine = KNNEngine.getEngine(in.readString());
        this.spaceType = SpaceType.getSpace(in.readString());
        this.dimension = in.readInt();
        this.state = new AtomicReference<>(ModelState.readFrom(in));
        this.timestamp = in.readString();

        // Description and error may be empty. However, reading the string will work as long as they are not null
        // which is checked in constructor and setters
        this.description = in.readString();
        ModelUtil.blockCommasInModelDescription(this.description);
        this.error = in.readString();

        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), IndexUtil.MODEL_NODE_ASSIGNMENT_KEY)) {
            this.trainingNodeAssignment = in.readString();
        } else {
            this.trainingNodeAssignment = "";
        }

        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), IndexUtil.MODEL_METHOD_COMPONENT_CONTEXT_KEY)) {
            this.methodComponentContext = new MethodComponentContext(in);
        } else {
            this.methodComponentContext = MethodComponentContext.EMPTY;
        }

        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY)) {
            this.vectorDataType = VectorDataType.get(in.readString());
        } else {
            this.vectorDataType = VectorDataType.DEFAULT;
        }

        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE)) {
            this.workloadModeConfig = WorkloadModeConfig.fromString(in.readOptionalString());
            this.compressionConfig = CompressionConfig.fromString(in.readOptionalString());
        } else {
            this.workloadModeConfig = WorkloadModeConfig.NOT_CONFIGURED;
            this.compressionConfig = CompressionConfig.NOT_CONFIGURED;
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(getKnnEngine().getName());
        out.writeString(getSpaceType().getValue());
        out.writeInt(getDimension());
        getState().writeTo(out);
        out.writeString(getTimestamp());
        out.writeString(getDescription());
        out.writeString(getError());
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), IndexUtil.MODEL_NODE_ASSIGNMENT_KEY)) {
            out.writeString(getNodeAssignment());
        }
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), IndexUtil.MODEL_METHOD_COMPONENT_CONTEXT_KEY)) {
            getMethodComponentContext().writeTo(out);
        }
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY)) {
            out.writeString(vectorDataType.getValue());
        }
        if (IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE)) {
            out.writeOptionalString(workloadModeConfig.toString());
            out.writeOptionalString(compressionConfig.toString());
        }
    }

    /**
     * Constructor
     *
     * @param knnEngine engine model is built with
     * @param spaceType space type model uses
     * @param dimension dimension of the model
     * @param modelState state of the model
     * @param timestamp timevalue when model was created
     * @param description information about the model
     * @param error error message associated with model
     * @param trainingNodeAssignment node assignment for the model
     * @param methodComponentContext method component context associated with model
     * @param vectorDataType vector data type of the model
     */
    public ModelMetadata(
        KNNEngine knnEngine,
        SpaceType spaceType,
        int dimension,
        ModelState modelState,
        String timestamp,
        String description,
        String error,
        String trainingNodeAssignment,
        MethodComponentContext methodComponentContext,
        VectorDataType vectorDataType,
        WorkloadModeConfig workloadModeConfig,
        CompressionConfig compressionConfig
    ) {
        this.knnEngine = Objects.requireNonNull(knnEngine, "knnEngine must not be null");
        this.spaceType = Objects.requireNonNull(spaceType, "spaceType must not be null");
        int maxDimensions = KNNEngine.getMaxDimensionByEngine(this.knnEngine);
        if (dimension <= 0 || dimension > maxDimensions) {
            throw new IllegalArgumentException(
                String.format(
                    "Dimension \"%s\" is invalid. Value must be greater than 0 and less than or equal to %d",
                    dimension,
                    maxDimensions
                )
            );
        }
        this.dimension = dimension;

        this.state = new AtomicReference<>(Objects.requireNonNull(modelState, "modelState must not be null"));
        this.timestamp = Objects.requireNonNull(timestamp, "timestamp must not be null");
        this.description = Objects.requireNonNull(description, "description must not be null");
        ModelUtil.blockCommasInModelDescription(this.description);
        this.error = Objects.requireNonNull(error, "error must not be null");
        this.trainingNodeAssignment = Objects.requireNonNull(trainingNodeAssignment, "node assignment must not be null");
        this.methodComponentContext = Objects.requireNonNull(methodComponentContext, "method context must not be null");
        this.vectorDataType = Objects.requireNonNull(vectorDataType, "vector data type must not be null");
        this.workloadModeConfig = workloadModeConfig;
        this.compressionConfig = compressionConfig;
    }

    /**
     * getter for model's state
     *
     * @return state
     */
    public ModelState getState() {
        return state.get();
    }

    /**
     * getter for model's node assignment
     *
     * @return trainingNodeAssignment
     */
    public String getNodeAssignment() {
        return trainingNodeAssignment;
    }

    /**
     * setter for model's state
     *
     * @param state of the model
     */
    public synchronized void setState(ModelState state) {
        this.state.set(Objects.requireNonNull(state, "state must not be null"));
    }

    /**
     * setter for model's error
     *
     * @param error set on failure
     */
    public synchronized void setError(String error) {
        this.error = error;
    }

    @Override
    public String toString() {
        return String.join(
            DELIMITER,
            knnEngine.getName(),
            spaceType.getValue(),
            Integer.toString(dimension),
            getState().toString(),
            timestamp,
            description,
            error,
            trainingNodeAssignment,
            methodComponentContext.toClusterStateString(),
            vectorDataType.getValue(),
            workloadModeConfig.toString(),
            compressionConfig.toString()
        );
    }

    /**
     * Returns ModelMetadata from string representation
     *
     * @param modelMetadataString String to be parsed
     * @return modelMetadata from string
     */
    public static ModelMetadata fromString(String modelMetadataString) {
        String[] modelMetadataArray = modelMetadataString.split(DELIMITER, -1);
        int length = modelMetadataArray.length;

        if (length < 7 || length > 12) {
            throw new IllegalArgumentException(
                "Illegal format for model metadata. Must be of the form "
                    + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>\" or "
                    + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>,<NodeAssignment>\" or "
                    + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>,<NodeAssignment>,<MethodContext>\" or "
                    + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>,<NodeAssignment>,<MethodContext>,<VectorDataType>\". or"
                    + "\"<KNNEngine>,<SpaceType>,<Dimension>,<ModelState>,<Timestamp>,<Description>,<Error>,<NodeAssignment>,<MethodContext>,<VectorDataType>,<WorkloadConfig>,<CompressionConfig>\"."
            );
        }

        KNNEngine knnEngine = KNNEngine.getEngine(modelMetadataArray[0]);
        SpaceType spaceType = SpaceType.getSpace(modelMetadataArray[1]);
        int dimension = Integer.parseInt(modelMetadataArray[2]);
        ModelState modelState = ModelState.getModelState(modelMetadataArray[3]);
        String timestamp = modelMetadataArray[4];
        String description = modelMetadataArray[5];
        String error = modelMetadataArray[6];
        String trainingNodeAssignment = length > 7 ? modelMetadataArray[7] : "";
        MethodComponentContext methodComponentContext = length > 8
            ? MethodComponentContext.fromClusterStateString(modelMetadataArray[8])
            : MethodComponentContext.EMPTY;
        VectorDataType vectorDataType = length > 9 ? VectorDataType.get(modelMetadataArray[9]) : VectorDataType.DEFAULT;
        WorkloadModeConfig workloadModeConfig = length > 10
            ? WorkloadModeConfig.fromString(modelMetadataArray[10])
            : WorkloadModeConfig.NOT_CONFIGURED;
        CompressionConfig compressionConfig = length > 11
            ? CompressionConfig.fromString(modelMetadataArray[11])
            : CompressionConfig.NOT_CONFIGURED;

        log.debug(getLogMessage(length));

        return new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            modelState,
            timestamp,
            description,
            error,
            trainingNodeAssignment,
            methodComponentContext,
            vectorDataType,
            workloadModeConfig,
            compressionConfig
        );
    }

    private static String getLogMessage(int length) {
        switch (length) {
            case 7:
                return "Model metadata array does not contain training node assignment or method component context. Assuming empty string node assignment and empty method component context.";
            case 8:
                return "Model metadata contains training node assignment. Assuming empty method component context.";
            case 9:
                return "Model metadata contains training node assignment and method context.";
            case 10:
                return "Model metadata contains training node assignment, method context and vector data type.";
            case 11:
            case 12:
                return "Model metadata contains workload mode config and compression config";
            default:
                throw new IllegalArgumentException("Unexpected metadata array length: " + length);
        }
    }

    private static String objectToString(Object value) {
        if (value == null) return null;
        return (String) value;
    }

    private static Integer objectToInteger(Object value) {
        if (value == null) return null;
        return (Integer) value;
    }

    /**
     * Returns ModelMetadata from Map representation
     *
     * @param modelSourceMap Map to be parsed
     * @return ModelMetadata instance
     */
    public static ModelMetadata getMetadataFromSourceMap(final Map<String, Object> modelSourceMap) {
        Object engine = modelSourceMap.get(KNNConstants.KNN_ENGINE);
        Object space = modelSourceMap.get(KNNConstants.METHOD_PARAMETER_SPACE_TYPE);
        Object dimension = modelSourceMap.get(KNNConstants.DIMENSION);
        Object state = modelSourceMap.get(KNNConstants.MODEL_STATE);
        Object timestamp = modelSourceMap.get(KNNConstants.MODEL_TIMESTAMP);
        Object description = modelSourceMap.get(KNNConstants.MODEL_DESCRIPTION);
        Object error = modelSourceMap.get(KNNConstants.MODEL_ERROR);
        Object trainingNodeAssignment = modelSourceMap.get(KNNConstants.MODEL_NODE_ASSIGNMENT);
        Object methodComponentContext = modelSourceMap.get(KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT);
        Object vectorDataType = modelSourceMap.get(KNNConstants.VECTOR_DATA_TYPE_FIELD);
        Object workloadModeConfig = modelSourceMap.get(KNNConstants.MODE_PARAMETER);
        Object compressionConfig = modelSourceMap.get(KNNConstants.COMPRESSION_PARAMETER);

        if (trainingNodeAssignment == null) {
            trainingNodeAssignment = "";
        }

        if (Objects.nonNull(methodComponentContext)) {
            try {
                XContentParser xContentParser = JsonXContent.jsonXContent.createParser(
                    NamedXContentRegistry.EMPTY,
                    IGNORE_DEPRECATIONS,
                    objectToString(methodComponentContext)
                );
                methodComponentContext = MethodComponentContext.fromXContent(xContentParser);
            } catch (IOException e) {
                throw new IllegalArgumentException("Error parsing method component context");
            }
        } else {
            methodComponentContext = MethodComponentContext.EMPTY;
        }

        if (vectorDataType == null) {
            vectorDataType = VectorDataType.DEFAULT.getValue();
        }

        return new ModelMetadata(
            KNNEngine.getEngine(objectToString(engine)),
            SpaceType.getSpace(objectToString(space)),
            objectToInteger(dimension),
            ModelState.getModelState(objectToString(state)),
            objectToString(timestamp),
            objectToString(description),
            objectToString(error),
            objectToString(trainingNodeAssignment),
            (MethodComponentContext) methodComponentContext,
            VectorDataType.get(objectToString(vectorDataType)),
            WorkloadModeConfig.fromString(workloadModeConfig == null ? null : workloadModeConfig.toString()),
            CompressionConfig.fromString(compressionConfig == null ? null : compressionConfig.toString())
        );
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(KNNConstants.MODEL_STATE, getState().getName());
        builder.field(KNNConstants.MODEL_TIMESTAMP, getTimestamp());
        builder.field(KNNConstants.MODEL_DESCRIPTION, getDescription());
        builder.field(KNNConstants.MODEL_ERROR, getError());

        builder.field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, getSpaceType().getValue());
        builder.field(KNNConstants.DIMENSION, getDimension());
        builder.field(KNNConstants.KNN_ENGINE, getKnnEngine().getName());
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion(IndexUtil.MODEL_NODE_ASSIGNMENT_KEY)) {
            builder.field(KNNConstants.MODEL_NODE_ASSIGNMENT, getNodeAssignment());
        }
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion(IndexUtil.MODEL_METHOD_COMPONENT_CONTEXT_KEY)) {
            builder.field(KNNConstants.MODEL_METHOD_COMPONENT_CONTEXT).startObject();
            getMethodComponentContext().toXContent(builder, params);
            builder.endObject();
        }
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion(KNNConstants.MODEL_VECTOR_DATA_TYPE_KEY)) {
            builder.field(KNNConstants.VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());
        }
        if (IndexUtil.isClusterOnOrAfterMinRequiredVersion(KNNConstants.MINIMAL_MODE_AND_COMPRESSION_FEATURE)) {
            if (workloadModeConfig != WorkloadModeConfig.NOT_CONFIGURED) {
                builder.field(KNNConstants.MODE_PARAMETER, workloadModeConfig.toString());
            }
            if (compressionConfig != CompressionConfig.NOT_CONFIGURED) {
                builder.field(KNNConstants.COMPRESSION_PARAMETER, compressionConfig.toString());
            }
        }
        return builder;
    }
}
