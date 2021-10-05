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
import org.opensearch.action.delete.DeleteResponse;

public class DeleteModelAction extends ActionType<DeleteResponse> {


    public static final DeleteModelAction INSTANCE = new DeleteModelAction();
    public static final String NAME = "cluster:admin/knn_delete_model_action";

    private DeleteModelAction() {
        super(NAME, DeleteResponse::new);
    }
}
