package com.drdo.anc;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
public record Run(@NotBlank String name,
                  @NotBlank String method,
                  double snr,
                  @JsonProperty("si_snr") double siSnr,
                  double ssnr,
                  double stoi,
                  double pesq,
                  @JsonProperty("latency_ms") double latencyMs,
                  double rtf,
                  @JsonProperty("commit") String commit,
                  @JsonProperty("date") String date) {
}
