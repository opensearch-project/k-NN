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
import org.opensearch.indices.IndicesService;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.training.TrainingJob;
import org.opensearch.knn.training.TrainingJobRunner;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import java.io.IOException;

/**
 * Transport action that trains a model and serializes it to model system index
 */
public class TrainingModelTransportAction extends HandledTransportAction<TrainingModelRequest, TrainingModelResponse> {

    private final IndicesService indicesService;

    @Inject
    public TrainingModelTransportAction(TransportService transportService,
                                        ActionFilters actionFilters,
                                        IndicesService indicesService) {
        super(TrainingModelAction.NAME, transportService, actionFilters, TrainingModelRequest::new);
        this.indicesService = indicesService;
    }

    @Override
    protected void doExecute(Task task, TrainingModelRequest request,
                             ActionListener<TrainingModelResponse> listener) {
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext =
                new NativeMemoryEntryContext.TrainingDataEntryContext(
                        request.getTrainingDataSizeInKB(),
                        request.getTrainingIndex(),
                        request.getTrainingField(),
                        NativeMemoryLoadStrategy.TrainingLoadStrategy.getInstance(),
                        indicesService,
                        request.getMaximumVectorCount(),
                        request.getSearchSize()
                );

        TrainingJob trainingJob = new TrainingJob(
                request.getModelId(),
                request.getKnnMethodContext(),
                NativeMemoryCacheManager.getInstance(),
                trainingDataEntryContext,
                request.getDimension(),
                request.getDescription()
        );

        try {
            TrainingJobRunner.getInstance().execute(trainingJob, ActionListener.wrap(
                    indexResponse -> listener.onResponse(new TrainingModelResponse(indexResponse.getId())),
                    listener::onFailure)
            );
        } catch (IOException e) {
            listener.onFailure(e);
        }
    }
}
