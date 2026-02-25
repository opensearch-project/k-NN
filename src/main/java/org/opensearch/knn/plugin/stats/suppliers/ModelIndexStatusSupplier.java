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

package org.opensearch.knn.plugin.stats.suppliers;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.knn.indices.ModelDao;

import java.util.function.Function;
import java.util.function.Supplier;

public class ModelIndexStatusSupplier<T> implements Supplier<T> {
    public static Logger logger = LogManager.getLogger(ModelIndexStatusSupplier.class);
    private Function<ModelDao, T> getter;

    /**
     * Constructor
     *
     * @param getter ModelDAO Method to supply a value
     */
    public ModelIndexStatusSupplier(Function<ModelDao, T> getter) {
        this.getter = getter;
    }

    @Override
    public T get() {
        try {
            return getter.apply(ModelDao.OpenSearchKNNModelDao.getInstance());
        } catch (ResourceNotFoundException e) { // catch to prevent exception to be raised.
            logger.info(e.getMessage());
            return null; // to let consumer knows that no value is available for getter.
        }
    }
}
