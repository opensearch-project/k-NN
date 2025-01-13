/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.junit.Before;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.common.settings.Settings;
import org.opensearch.test.rest.OpenSearchRestTestCase;

import java.util.Locale;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.*;

public abstract class AbstractRollingUpgradeTestCase extends KNNRestTestCase {
    protected String testIndex;

    @Before
    protected void setIndex() {
        // Creating index name by concatenating "knn-bwc-" prefix with test method name
        // for all the tests in this sub-project
        testIndex = KNN_BWC_PREFIX + getTestName().toLowerCase(Locale.ROOT);
    }

    @Override
    protected final boolean preserveIndicesUponCompletion() {
        return true;
    }

    @Override
    protected final boolean preserveReposUponCompletion() {
        return true;
    }

    @Override
    protected boolean preserveTemplatesUponCompletion() {
        return true;
    }

    @Override
    protected final Settings restClientSettings() {
        return Settings.builder()
            .put(super.restClientSettings())
            // increase the timeout here to 90 seconds to handle long waits for a green
            // cluster health. the waits for green need to be longer than a minute to
            // account for delayed shards
            .put(OpenSearchRestTestCase.CLIENT_SOCKET_TIMEOUT, CLIENT_TIMEOUT_VALUE)
            .build();
    }

    protected enum ClusterType {
        OLD,
        MIXED,
        UPGRADED;

        public static ClusterType instance(String value) {
            switch (value) {
                case OLD_CLUSTER:
                    return OLD;
                case MIXED_CLUSTER:
                    return MIXED;
                case UPGRADED_CLUSTER:
                    return UPGRADED;
                default:
                    throw new IllegalArgumentException("unknown cluster type: " + value);
            }
        }
    }

    protected final ClusterType getClusterType() {
        if (System.getProperty(BWCSUITE_CLUSTER) == null) {
            throw new IllegalArgumentException(String.format("[%s] value is null", BWCSUITE_CLUSTER));
        }
        return ClusterType.instance(System.getProperty(BWCSUITE_CLUSTER));
    }

    protected final boolean isFirstMixedRound() {
        return Boolean.parseBoolean(System.getProperty(ROLLING_UPGRADE_FIRST_ROUND, "false"));
    }

    protected final Optional<String> getBWCVersion() {
        return Optional.ofNullable(System.getProperty(BWC_VERSION, null));
    }

    @Override
    protected Settings getKNNDefaultIndexSettings() {
        if (isApproximateThresholdSupported(getBWCVersion())) {
            return super.getKNNDefaultIndexSettings();
        }
        // for bwc will return old default setting without approximate value threshold setting
        return getDefaultIndexSettings();
    }

}
