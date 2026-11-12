# Experiment 4: dictionary attack on PPFLE (test split)

Levels are cumulative (L1: enumerate low-cardinality columns; L2: + brute-force integers). % rows = rows whose every token is recovered. A recovered E1 token is the exact value; a recovered E2 token is the bin index (a quantile range of the private train data). Every recovered token was checked against the true value (0 wrong).

| Dataset | Encoding | enumerate: % tokens | enumerate: % rows | bruteforce: % tokens | bruteforce: % rows |
|---|---|---:|---:|---:|---:|
| Edge-IIoTset (fixed) | E1 PPFLE | 43.4 | 0.0 | 97.8 | 18.9 |
| Edge-IIoTset (fixed) | E2 PPFLE-bins | 99.9 | 96.0 | 99.9 | 96.0 |
| Edge-IIoTset (as published) | E1 PPFLE | 43.4 | 0.0 | 97.9 | 19.5 |
| Edge-IIoTset (as published) | E2 PPFLE-bins | 99.9 | 96.0 | 99.9 | 96.0 |

## Share of test tokens recovered per column (pooled over datasets, weighted by rows)

| Column | E1 PPFLE enumerate | E1 PPFLE bruteforce | E2 PPFLE-bins enumerate | E2 PPFLE-bins bruteforce |
|---|---:|---:|---:|---:|
| arp.opcode | 100.0 | 100.0 | 100.0 | 100.0 |
| arp.hw.size | 0.0 | 100.0 | 100.0 | 100.0 |
| icmp.checksum | 0.0 | 100.0 | 100.0 | 100.0 |
| icmp.seq_le | 0.0 | 100.0 | 100.0 | 100.0 |
| icmp.unused | 0.0 | 100.0 | 100.0 | 100.0 |
| http.content_length | 0.0 | 100.0 | 100.0 | 100.0 |
| http.request.method | 100.0 | 100.0 | 100.0 | 100.0 |
| http.referer | 100.0 | 100.0 | 100.0 | 100.0 |
| http.request.version | 100.0 | 100.0 | 100.0 | 100.0 |
| http.response | 0.0 | 100.0 | 100.0 | 100.0 |
| http.tls_port | 0.0 | 100.0 | 100.0 | 100.0 |
| tcp.ack | 0.0 | 94.0 | 100.0 | 100.0 |
| tcp.ack_raw | 0.0 | 23.1 | 100.0 | 100.0 |
| tcp.checksum | 0.0 | 100.0 | 100.0 | 100.0 |
| tcp.connection.fin | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.connection.rst | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.connection.syn | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.connection.synack | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.flags | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.flags.ack | 100.0 | 100.0 | 100.0 | 100.0 |
| tcp.len | 0.0 | 100.0 | 100.0 | 100.0 |
| tcp.seq | 0.0 | 91.3 | 100.0 | 100.0 |
| udp.stream | 0.0 | 96.6 | 100.0 | 100.0 |
| udp.time_delta | 0.0 | 100.0 | 100.0 | 100.0 |
| dns.qry.name | 0.0 | 99.6 | 100.0 | 100.0 |
| dns.qry.name.len | 99.8 | 99.8 | 99.8 | 99.8 |
| dns.qry.qu | 0.0 | 100.0 | 100.0 | 100.0 |
| dns.qry.type | 100.0 | 100.0 | 100.0 | 100.0 |
| dns.retransmission | 0.0 | 100.0 | 100.0 | 100.0 |
| dns.retransmit_request | 0.0 | 100.0 | 100.0 | 100.0 |
| dns.retransmit_request_in | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.conack.flags | 100.0 | 100.0 | 100.0 | 100.0 |
| mqtt.conflag.cleansess | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.conflags | 100.0 | 100.0 | 100.0 | 100.0 |
| mqtt.hdrflags | 100.0 | 100.0 | 100.0 | 100.0 |
| mqtt.len | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.msg_decoded_as | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.msgtype | 100.0 | 100.0 | 100.0 | 100.0 |
| mqtt.proto_len | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.protoname | 100.0 | 100.0 | 100.0 | 100.0 |
| mqtt.topic | 96.3 | 96.3 | 96.3 | 96.3 |
| mqtt.topic_len | 0.0 | 100.0 | 100.0 | 100.0 |
| mqtt.ver | 100.0 | 100.0 | 100.0 | 100.0 |
| mbtcp.len | 0.0 | 100.0 | 100.0 | 100.0 |
| mbtcp.trans_id | 0.0 | 100.0 | 100.0 | 100.0 |
| mbtcp.unit_id | 100.0 | 100.0 | 100.0 | 100.0 |
