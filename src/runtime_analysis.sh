source venv/bin/activate

for S in 2 4 8 16 32
do
    for D in 1 5
    do
        python3 fs_benchmark.py \
            --nosim \
            --nomem \
            --dataset bvl \
            --runs 5 \
            --coupling 100grid \
            --output-file test_bvl_S\=$S\_D=$D.csv \
            --slack "$D" \
            --solncap "$S"
        python3 fs_benchmark.py \
            --nosim \
            --mem \
            --dataset bvl \
            --runs 1 \
            --coupling 100grid \
            --output-file test_bvl_mem_S\=$S\_D=$D.csv \
            --slack "$D" \
            --solncap "$S"
    done
done

deactivate
