#!/bin/bash
shopt -s nullglob
for fchk in *.fchk; do
    name=$(basename "${fchk%.fchk}")
    echo "Creating and running ${name}.hs"
    cat > "${name}.hs" <<EOT
\$READFCHK
${fchk}
\$NORINGS
\$PARTITION
ALL
EOT
    esipy "${name}.hs" > "${name}.esi"
done
