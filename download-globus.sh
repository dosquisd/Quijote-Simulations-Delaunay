set -a
source .env
set +a

simulation=$1
method=$2

if [ -z "$simulation" ]; then
    echo "Usage: $0 <simulation> <method>"
    exit 1
fi

if [ -z "$method" ]; then
    echo "No method specified, defaulting to 'standard'"
    method="standard"
fi

if [ "$method" != "standard" ] && [ "$method" != "paired_fixed" ]; then
    echo "Error: method must be either 'standard' or 'paired_fixed'"
    exit 1
fi

QUIJOTE_PATH="/Halos/FoF"
PC_PATH="$(pwd)/quijote/FoF"

if [ "$method" == "standard" ]; then
    for i in {0..9}; do
        echo "Downloading simulation $simulation, realization $i, method $method"
        globus transfer $GLOBUS_QUIJOTEID:${QUIJOTE_PATH}/${simulation}/${i} $GLOBUS_PCID:${PC_PATH}/${simulation}/${i} --recursive
    done
else
    for j in 0 1; do
        for i in {0..4}; do
            echo "Downloading simulation $simulation, realization $i, paired fixed $j"
            globus transfer $GLOBUS_QUIJOTEID:${QUIJOTE_PATH}/${simulation}/NCV_${j}_${i} $GLOBUS_PCID:${PC_PATH}/${simulation}/NCV_${j}_${i} --recursive
        done
    done
fi
