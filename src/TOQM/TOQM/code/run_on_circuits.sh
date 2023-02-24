benchmarks_base="../../../../benchmarks"
for circuit_name in "4_49_16.qasm" "wim_266.qasm" "square_root_7.qasm" "hwb4_49.qasm" "vqe_n8.qasm" "cycle10_2_110.qasm" "adr4_197.qasm" "sym9_148.qasm" "life_238.qasm"
do
    echo "circuit: ${circuit_name}"
    output_circ="${benchmarks_base}/mapped_circuits/google_sycamore/${circuit_name}/toqm_circ.qasm"
    time ./mapper "${benchmarks_base}/mapped_circuits/google_sycamore/${circuit_name}/base_mapping.qasm" "arch/google_sycamore.arch" -defaults -latency Latency_1_2_6 -expander GreedyTopK 10 -queue TrimSlowNodes 2000 1000 -nodeMod GreedyMapper -retain 1 > $output_circ
    sed -i '' "s/swp/swap/g" $output_circ
done
