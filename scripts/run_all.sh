#!/bin/bash
# Run all figure scripts in order
set -e
cd "$(dirname "$0")"
echo "===================================================="
echo "Generating ALL figures from deduplicated Table 2"
echo "===================================================="
for s in fig1_distributions fig2_fits fig3_coupling fig4_morphology \
         fig5_physical fig6_ratio fig7_mstar fig8_tng50; do
    echo
    echo ">>> $s"
    python3 ${s}.py
done
echo
echo "===================================================="
echo "DONE. All figures in: ../figures/"
echo "===================================================="
ls -la ../figures/
