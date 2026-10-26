package com.drdo.anc;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.ToDoubleFunction;

public class RunSorter {
    private static final Map<String, ToDoubleFunction<Run>> METRICS = Map.of(
            "snr", Run::snr, "si_snr", Run::siSnr, "ssnr", Run::ssnr, "stoi", Run::stoi,
            "pesq", Run::pesq, "latency_ms", Run::latencyMs, "rtf", Run::rtf);
    private static final Set<String> LOWER_IS_BETTER = Set.of("latency_ms", "rtf");

    private RunSorter() {
    }

    /** True if the name is a sortable metric. */
    public static boolean isMetric(String name) {
        return METRICS.containsKey(name);
    }

    /** Sort by metric; a null order means best first (asc for latency and rtf). */
    public static List<Run> sort(List<Run> runs, String sortBy, String order) {
        Comparator<Run> cmp = Comparator.comparingDouble(METRICS.get(sortBy));
        boolean asc = order == null ? LOWER_IS_BETTER.contains(sortBy) : order.equals("asc");
        return runs.stream().sorted(asc ? cmp : cmp.reversed()).toList();
    }

    /** Name of the best run for every metric. */
    public static Map<String, String> best(List<Run> runs) {
        Map<String, String> result = new TreeMap<>();
        for (String metric : METRICS.keySet()) {
            Comparator<Run> cmp = Comparator.comparingDouble(METRICS.get(metric));
            runs.stream()
                    .max(LOWER_IS_BETTER.contains(metric) ? cmp.reversed() : cmp)
                    .ifPresent(r -> result.put(metric, r.name()));
        }
        return result;
    }
}
