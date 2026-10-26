package com.drdo.anc;

import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

/** POST and GET /api/runs, plus compare. */
@RestController
@RequestMapping("/api/runs")
public class RunController {
    static final Set<String> METHODS =
            Set.of("noisy", "spectral", "wiener", "lms", "nlms", "ai", "hybrid");
    private final RunRepository repo;

    public RunController(RunRepository repo) {
        this.repo = repo;
    }

    /** Save a run; 400 if method is not one of the contract values. */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Run create(@Valid @RequestBody Run run) {
        if (!METHODS.contains(run.method())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "invalid method");
        }
        repo.insert(run);
        return run;
    }

    /** List runs; optional sortBy=<metric> and order=asc|desc. */
    @GetMapping
    public List<Run> list(
            @RequestParam(name = "sortBy", required = false) String sortBy,
            @RequestParam(name = "order", required = false) String order) {
        List<Run> runs = repo.findAll();
        if (sortBy == null) {
            return runs;
        }
        boolean badOrder = order != null && !order.equals("asc") && !order.equals("desc");
        if (!RunSorter.isMetric(sortBy) || badOrder) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "invalid sortBy or order");
        }
        return RunSorter.sort(runs, sortBy, order);
    }

    /** Compare runs by name: returns the runs and the best run per metric. */
    @GetMapping("/compare")
    public Map<String, Object> compare(@RequestParam(name = "names") List<String> names) {
        List<Run> picked = repo.findAll().stream()
                .filter(r -> names.contains(r.name())).toList();
        if (picked.size() < 2) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "need two or more runs");
        }
        return Map.of("runs", picked, "best", RunSorter.best(picked));
    }
}