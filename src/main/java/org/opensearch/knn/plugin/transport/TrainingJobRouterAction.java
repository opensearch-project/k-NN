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

/**
 * Action to route training request to a particular node in the cluster
 */
public class TrainingJobRouterAction extends ActionType<TrainingModelResponse> {

    public static final String NAME = "cluster:admin/knn_training_job_router_action";
    public static final TrainingJobRouterAction INSTANCE = new TrainingJobRouterAction(NAME, TrainingModelResponse::new);

    private TrainingJobRouterAction(String name, Writeable.Reader<TrainingModelResponse> trainingModelResponseReader) {
        super(name, trainingModelResponseReader);
    }
}
