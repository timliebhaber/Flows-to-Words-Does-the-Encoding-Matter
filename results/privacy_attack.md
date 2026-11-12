# Experiment 4: dictionary attack on PPFLE (test split)

Levels are cumulative (L1: enumerate low-cardinality columns; L2: + brute-force integers; L3: + public-data dictionary). % rows = rows whose every token is recovered. A recovered E1 token is the exact value; a recovered E2 token is the bin index (a quantile range of the private train data). Every recovered token was checked against the true value (0 wrong).

| Dataset | Encoding | enumerate: % tokens | enumerate: % rows | bruteforce: % tokens | bruteforce: % rows | public: % tokens | public: % rows |
|---|---|---:|---:|---:|---:|---:|---:|
| UNSW-NB15 | E1 PPFLE | 25.6 | 0.0 | 94.7 | 0.3 | 95.0 | 3.8 |
| UNSW-NB15 | E2 PPFLE-bins | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ToN-IoT | E1 PPFLE | 25.5 | 0.0 | 99.7 | 87.6 | 99.9 | 93.5 |
| ToN-IoT | E2 PPFLE-bins | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| BoT-IoT | E1 PPFLE | 25.9 | 0.0 | 97.3 | 7.7 | 97.5 | 11.4 |
| BoT-IoT | E2 PPFLE-bins | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| CIC-IDS2018 | E1 PPFLE | 25.5 | 0.0 | 99.8 | 93.1 | 99.9 | 95.7 |
| CIC-IDS2018 | E2 PPFLE-bins | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Share of test tokens recovered per column (pooled over datasets, weighted by rows)

| Column | E1 PPFLE enumerate | E1 PPFLE bruteforce | E1 PPFLE public | E2 PPFLE-bins enumerate | E2 PPFLE-bins bruteforce | E2 PPFLE-bins public |
|---|---:|---:|---:|---:|---:|---:|
| L4_DST_PORT | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| PROTOCOL | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| L7_PROTO | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| IN_BYTES | 0.0 | 98.8 | 98.8 | 100.0 | 100.0 | 100.0 |
| IN_PKTS | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| OUT_BYTES | 0.0 | 99.1 | 99.1 | 100.0 | 100.0 | 100.0 |
| OUT_PKTS | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| TCP_FLAGS | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| CLIENT_TCP_FLAGS | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SERVER_TCP_FLAGS | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| FLOW_DURATION_MILLISECONDS | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DURATION_IN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DURATION_OUT | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| MIN_TTL | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| MAX_TTL | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| LONGEST_FLOW_PKT | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SHORTEST_FLOW_PKT | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| MIN_IP_PKT_LEN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| MAX_IP_PKT_LEN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_SECOND_BYTES | 2.4 | 68.4 | 69.9 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_SECOND_BYTES | 2.4 | 51.2 | 52.2 | 100.0 | 100.0 | 100.0 |
| RETRANSMITTED_IN_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RETRANSMITTED_IN_PKTS | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RETRANSMITTED_OUT_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| RETRANSMITTED_OUT_PKTS | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_AVG_THROUGHPUT | 0.0 | 91.2 | 94.0 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_AVG_THROUGHPUT | 0.0 | 86.7 | 88.8 | 100.0 | 100.0 | 100.0 |
| NUM_PKTS_UP_TO_128_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NUM_PKTS_128_TO_256_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NUM_PKTS_256_TO_512_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NUM_PKTS_512_TO_1024_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| NUM_PKTS_1024_TO_1514_BYTES | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| TCP_WIN_MAX_IN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| TCP_WIN_MAX_OUT | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ICMP_TYPE | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ICMP_IPV4_TYPE | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DNS_QUERY_TYPE | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DNS_TTL_ANSWER | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| FTP_COMMAND_RET_CODE | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_IAT_MIN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_IAT_MAX | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_IAT_AVG | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| SRC_TO_DST_IAT_STDDEV | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_IAT_MIN | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_IAT_MAX | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_IAT_AVG | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| DST_TO_SRC_IAT_STDDEV | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
