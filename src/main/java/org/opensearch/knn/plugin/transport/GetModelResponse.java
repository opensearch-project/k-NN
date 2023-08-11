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
package org.opensearch.knn.plugin.transport;

import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.indices.Model;

import java.io.IOException;

/**
 * {@link GetModelResponse} represents Response returned by {@link GetModelRequest}
 */
public class GetModelResponse extends ActionResponse implements ToXContentObject {

    private final Model model;

    public GetModelResponse(Model model) {
        this.model = model;
    }

    public GetModelResponse(StreamInput in) throws IOException {
        super(in);
        this.model = new Model(in);
    }

    public Model getModel() {
        return model;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        /* Response should look like below:
            {
                "model_id": "my_model_id"
                "state": "created",
                "created_timestamp": "10-31-21 02:02:02",
                "description": "Model trained with dataset X",
                "error": "",
                "model_blob": "cdscsacsadcsdca",
                "engine": "faiss",
                "space_type": "l2",
                "dimension": 128
        }
         */
        return model.toXContent(builder, params);
    }

    @Override
    public void writeTo(StreamOutput output) throws IOException {
        model.writeTo(output);
    }
}
