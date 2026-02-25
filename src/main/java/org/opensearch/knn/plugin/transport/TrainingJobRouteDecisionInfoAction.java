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
 * Action used to collect information from each node to determine which node would be best to route a particular
 * training job to.
 */
public class TrainingJobRouteDecisionInfoAction extends ActionType<TrainingJobRouteDecisionInfoResponse> {

    public static final String NAME = "cluster:admin/knn_training_job_route_decision_info_action";
    public static final TrainingJobRouteDecisionInfoAction INSTANCE = new TrainingJobRouteDecisionInfoAction(
        NAME,
        TrainingJobRouteDecisionInfoResponse::new
    );

    /**
     * Constructor.
     *
     * @param name name of action
     * @param responseReader reader for TrainingJobRouteDecisionInfoResponse response
     */
    public TrainingJobRouteDecisionInfoAction(String name, Writeable.Reader<TrainingJobRouteDecisionInfoResponse> responseReader) {
        super(name, responseReader);
    }
}
