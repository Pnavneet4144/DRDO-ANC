package com.drdo.anc;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.web.server.ResponseStatusException;

@SpringBootTest
public class RunControllerTest {
    @Autowired
    RunController controller;

    // TEST FIXTURE ONLY: dummy values, not a measured result.
    private Run fixture(String method) {
        return new Run("test_fixture", method, 0, 0, 0, 0, 0, 0, 0, "abc123", "2026-01-01");
    }

    @Test
    void roundTrip() {
        controller.create(fixture("nlms"));
        assertTrue(controller.list(null,null).contains(fixture("nlms")));
    }

    @Test
    void invalidMethodRejected() {
        assertThrows(ResponseStatusException.class, () -> controller.create(fixture("bad")));
    }
    @Test
    void invalidSortByRejected() {
        assertThrows(ResponseStatusException.class, () -> controller.list("bad", null));
    }
}
