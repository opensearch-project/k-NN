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

import org.opensearch.common.Nullable;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.Base64;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

import static org.opensearch.knn.common.KNNConstants.MODEL_BLOB_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;

public class Model implements Writeable, ToXContentObject {

    private String modelID;
    private ModelMetadata modelMetadata;
    private AtomicReference<byte[]> modelBlob;

    /**
     * Constructor
     *
     * @param modelMetadata metadata about the model
     * @param modelBlob binary representation of model template index. Can be null if model is not yet in CREATED state.
     * @param modelID model identifier
     */
    public Model(ModelMetadata modelMetadata, @Nullable byte[] modelBlob, String modelID) {
        this.modelMetadata = Objects.requireNonNull(modelMetadata, "modelMetadata must not be null");

        if (ModelState.CREATED.equals(this.modelMetadata.getState()) && modelBlob == null) {
            throw new IllegalArgumentException(
                "Cannot construct model in state CREATED when model binary is null. " + "State must be either TRAINING or FAILED"
            );
        }

        this.modelBlob = new AtomicReference<>(modelBlob);
        this.modelID = Objects.requireNonNull(modelID, "model id must not be null");
    }

    private byte[] readOptionalModelBlob(StreamInput in) throws IOException {
        return in.readBoolean() ? in.readByteArray() : null;
    }

    /**
     * Constructor
     *
     * @param in Stream input
     */
    public Model(StreamInput in) throws IOException {
        this.modelMetadata = new ModelMetadata(in);
        this.modelBlob = new AtomicReference<>(readOptionalModelBlob(in));
        this.modelID = in.readString();
    }

    /**
     * getter for model's metadata
     *
     * @return model's metadata
     */
    public ModelMetadata getModelMetadata() {
        return modelMetadata;
    }

    /**
     * getter for model's identifier
     *
     * @return model's id
     */
    public String getModelID() {
        return modelID;
    }

    /**
     * getter for model's binary blob
     *
     * @return modelBlob
     */
    public byte[] getModelBlob() {
        return modelBlob.get();
    }

    /**
     * getter for model's length
     *
     * @return length of model blob
     */
    public int getLength() {
        if (getModelBlob() == null) {
            return 0;
        }
        return getModelBlob().length;
    }

    /**
     * Sets model blob to new value
     *
     * @param modelBlob updated model blob
     */
    public synchronized void setModelBlob(byte[] modelBlob) {
        this.modelBlob = new AtomicReference<>(Objects.requireNonNull(modelBlob, "model blob cannot be updated to null"));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Model other = (Model) obj;
        return other.getModelID().equals(this.getModelID());
    }

    @Override
    public int hashCode() {
        return getModelID().hashCode();
    }

    /**
     *  Parse source map content into {@link Model} instance.
     *
     * @param sourceMap source contents
     * @return model instance
     */
    public static Model getModelFromSourceMap(Map<String, Object> sourceMap) {
        String modelID = getModelIDFromResponse(sourceMap);
        ModelMetadata modelMetadata = ModelMetadata.getMetadataFromSourceMap(sourceMap);
        byte[] blob = getModelBlobFromResponse(sourceMap);
        return new Model(modelMetadata, blob, modelID);
    }

    private void writeOptionalModelBlob(StreamOutput output) throws IOException {
        if (getModelBlob() == null) {
            output.writeBoolean(false);
            return;
        }
        output.writeBoolean(true);
        output.writeByteArray(getModelBlob());
    }

    /**
     * Write this into the {@linkplain StreamOutput}.
     *
     * @param output instance of {@linkplain StreamOutput}.
     */
    @Override
    public void writeTo(StreamOutput output) throws IOException {
        getModelMetadata().writeTo(output);
        writeOptionalModelBlob(output);
        output.writeString(modelID);
    }

    private static String getModelIDFromResponse(Map<String, Object> responseMap) {
        Object modelId = responseMap.get(MODEL_ID);
        if (modelId == null) {
            return null;
        }
        return (String) modelId;
    }

    private static byte[] getModelBlobFromResponse(Map<String, Object> responseMap) {
        Object blob = responseMap.get(KNNConstants.MODEL_BLOB_PARAMETER);

        // If byte blob is not there, it means that the state has not yet been updated to CREATED.
        if (blob == null) {
            return null;
        }
        return Base64.getDecoder().decode((String) blob);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        XContentBuilder xContentBuilder = builder.startObject();
        builder.field(MODEL_ID, modelID);
        String base64Model = "";
        if (getModelBlob() != null) {
            base64Model = Base64.getEncoder().encodeToString(getModelBlob());
        }
        builder.field(MODEL_BLOB_PARAMETER, base64Model);
        getModelMetadata().toXContent(builder, params);
        return xContentBuilder.endObject();
    }
}
