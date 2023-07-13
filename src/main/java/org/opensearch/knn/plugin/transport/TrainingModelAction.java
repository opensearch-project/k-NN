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

import org.opensearch.action.ActionType;
import org.opensearch.core.common.io.stream.Writeable;

public class TrainingModelAction extends ActionType<TrainingModelResponse> {

    public static final String NAME = "cluster:admin/knn_training_model_action";
    public static final TrainingModelAction INSTANCE = new TrainingModelAction(NAME, TrainingModelResponse::new);

    private TrainingModelAction(String name, Writeable.Reader<TrainingModelResponse> trainingModelResponseReader) {
        super(name, trainingModelResponseReader);
    }
}
