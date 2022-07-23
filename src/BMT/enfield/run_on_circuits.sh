benchmark_path=../../../benchmarks

for circ_name in "hwb4_49.qasm" "wim_266.qasm" "misex1_241.qasm" "cycle10_2_110.qasm" "square_root_7.qasm" "vqe_n8.qasm" #"life_238.qasm"
do
    echo "circuit: ${circ_name}"
    circ_path="${benchmark_path}/mapped_circuits/ibm_tokyo/${circ_name}/base_mapping.qasm"
    output_path="${benchmark_path}/mapped_circuits/ibm_tokyo/${circ_name}/bmt_circ.qasm"
    time ./efd -i $circ_path -alloc Q_bmt -arch-file archfiles/tokyo.json -o $output_path
done

