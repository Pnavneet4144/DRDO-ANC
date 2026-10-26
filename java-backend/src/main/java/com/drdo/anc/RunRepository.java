package com.drdo.anc;
import java.util.List;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class RunRepository {
    private final JdbcTemplate jdbc;

    public RunRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** Insert one run. */
    public void insert(Run r) {
        jdbc.update("INSERT INTO runs (name, method, snr, si_snr, ssnr, stoi, pesq, latency_ms,"
                        + " rtf, commit_hash, run_date) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                r.name(), r.method(), r.snr(), r.siSnr(), r.ssnr(), r.stoi(), r.pesq(),
                r.latencyMs(), r.rtf(), r.commit(), r.date());
    }

    /** Return all runs, oldest first. */
    public List<Run> findAll() {
        return jdbc.query("SELECT * FROM runs ORDER BY id", (rs, i) -> new Run(
                rs.getString("name"), rs.getString("method"), rs.getDouble("snr"),
                rs.getDouble("si_snr"), rs.getDouble("ssnr"), rs.getDouble("stoi"),
                rs.getDouble("pesq"), rs.getDouble("latency_ms"), rs.getDouble("rtf"),
                rs.getString("commit_hash"), rs.getString("run_date")));
    }
}
