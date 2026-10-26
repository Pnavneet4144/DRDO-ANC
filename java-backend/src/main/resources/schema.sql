CREATE TABLE IF NOT EXISTS runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    method VARCHAR(20) NOT NULL,
    snr DOUBLE, si_snr DOUBLE, ssnr DOUBLE, stoi DOUBLE, pesq DOUBLE,
    latency_ms DOUBLE, rtf DOUBLE,
    commit_hash VARCHAR(64),
    run_date VARCHAR(10)
    );