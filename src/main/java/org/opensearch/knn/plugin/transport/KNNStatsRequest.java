/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.plugin.stats.StatNames;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

/**
 * KNNStatsRequest gets node (cluster) level Stats for KNN
 * By default, all parameters will be true
 */
public class KNNStatsRequest extends BaseNodesRequest<KNNStatsRequest> {

    /**
     * Key indicating all stats should be retrieved
     */
    public static final String ALL_STATS_KEY = "_all";
    private final Set<String> validStats;
    private final Set<String> statsToBeRetrieved;

    /**
     * Empty constructor needed for KNNStatsTransportAction
     */
    public KNNStatsRequest() {
        super((String[]) null);
        validStats = getValidStats();
        statsToBeRetrieved = new HashSet<>();
    }

    /**
     * Constructor
     *
     * @param in input stream
     * @throws IOException in case of I/O errors
     */
    public KNNStatsRequest(StreamInput in) throws IOException {
        super(in);
        validStats = in.readSet(StreamInput::readString);
        statsToBeRetrieved = in.readSet(StreamInput::readString);
    }

    /**
     * Constructor
     *
     * @param nodeIds NodeIDs from which to retrieve stats
     */
    public KNNStatsRequest(String... nodeIds) {
        super(nodeIds);
        validStats = getValidStats();
        statsToBeRetrieved = new HashSet<>();
    }

    /**
     * Add all stats to be retrieved
     */
    public void all() {
        statsToBeRetrieved.addAll(validStats);
    }

    /**
     * Remove all stats from retrieval set
     */
    public void clear() {
        statsToBeRetrieved.clear();
    }

    /**
     * Sets a stats retrieval status to true if it is a valid stat
     * @param stat stat name
     * @return true if the stats's retrieval status is successfully update; false otherwise
     */
    public boolean addStat(String stat) {
        if (validStats.contains(stat)) {
            statsToBeRetrieved.add(stat);
            return true;
        }
        return false;
    }

    /**
     * Get the set that tracks which stats should be retrieved
     *
     * @return the set that contains the stat names marked for retrieval
     */
    public Set<String> getStatsToBeRetrieved() {
        return statsToBeRetrieved;
    }

    /**
     * Get all valid stats, possibly omitting stats associated with disabled features
     */
    private Set<String> getValidStats() {
        Set<String> stats = StatNames.getNames();
        return stats;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeStringCollection(validStats);
        out.writeStringCollection(statsToBeRetrieved);
    }
}
