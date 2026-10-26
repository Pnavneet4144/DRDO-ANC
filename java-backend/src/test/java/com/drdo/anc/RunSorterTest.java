package com.drdo.anc;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;
public class RunSorterTest {
    // TEST FIXTURE ONLY: dummy values, not measured results.
    private Run run(String name, double stoi, double latency) {
        return new Run(name, "ai", 0, 0, 0, stoi, 0, latency, 0, "abc123", "2026-01-01");
    }

    private final List<Run> runs = List.of(run("a", 0.5, 30), run("b", 0.9, 10), run("c", 0.7, 20));

    @Test
    void higherIsBetterByDefault() {
        assertEquals("b", RunSorter.sort(runs, "stoi", null).get(0).name());
    }

    @Test
    void latencyDefaultsToLowestFirst() {
        assertEquals("b", RunSorter.sort(runs, "latency_ms", null).get(0).name());
    }

    @Test
    void explicitOrderOverridesDefault() {
        assertEquals("a", RunSorter.sort(runs, "stoi", "asc").get(0).name());
    }

    @Test
    void bestPerMetric() {
        assertEquals("b", RunSorter.best(runs).get("stoi"));
        assertEquals("b", RunSorter.best(runs).get("latency_ms"));
    }
}
