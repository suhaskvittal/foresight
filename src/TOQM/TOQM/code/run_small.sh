benchmarks_base="../../../../benchmarks"
for circuit_name in "4gt11_82.qasm" "4mod5-v0_19.qasm" "alu-v1_28.qasm" "bv5.qasm" "decod24-v2_43.qasm" 
do
    echo "circuit: ${circuit_name}"
    output_circ="${benchmarks_base}/solver_circuits/ibm_manila/${circuit_name}/toqm_circ.qasm"
    time ./mapper "${benchmarks_base}/solver_circuits_v2/${circuit_name}/base_mapping.qasm" "arch/ibm_manila.arch" -defaults -latency Latency_1_2_6 -filter HashFilter -filter HashFilter2 -pureSwapDiameter > $output_circ
    echo "cnots\t0\ndepth\t0\ntime\t0\nmemory\t0" > "${benchmarks_base}/solver_circuits/ibm_manila/${circuit_name}/toqm_data.txt"
    sed -i '' "s/swp/swap/g" $output_circ
done
