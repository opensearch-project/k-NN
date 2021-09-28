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

import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

public class TrainingModelTransportAction extends HandledTransportAction<TrainingModelRequest, TrainingModelResponse> {


    @Inject
    public TrainingModelTransportAction(String actionName, TransportService transportService,
                                           ActionFilters actionFilters,
                                           Writeable.Reader<TrainingModelRequest> trainModelRequestReader) {
        super(actionName, transportService, actionFilters, trainModelRequestReader);
    }

    @Override
    protected void doExecute(Task task, TrainingModelRequest request, ActionListener<TrainingModelResponse> actionListener) {

    }
}
