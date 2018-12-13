#!/usr/bin/env bash
LANG=en_us_88591

# Get the arguments specified on the command line
while getopts s:d:e: option
do
    case "${option}" in
       s) START_DATE=${OPTARG};; # Date Format: YYYY-MM-DD (Default: Today's date)
       d) DATE_DIFF=${OPTARG};; # Specifies how many days to harvest at a time (Default: 2)
       e) STOP_DATE=${OPTARG};; # Date Format: YYYY-MM-DD (Default: Today's date)
       \? ) echo "Unknown option: -$OPTARG" >&2; exit 1;;
       :  ) echo "Missing option argument for -$OPTARG" >&2; exit 1;;
       *  ) echo "Unimplemented option: -$OPTARG" >&2; exit 1;;
    esac
done

# Set default arguments to use if not specified on the command line
if [ ${OPTIND} == 1 ]
then
    STOP_DATE=$(date)
    STOP_DATE=$(date -d"$STOP_DATE" +%Y-%m-%d)
    DATE_DIFF=1
    START_DATE=$(date -d"${STOP_DATE} - $DATE_DIFF day" +%Y-%m-%d)
fi

# Convert dates to integer like format for easier comparisons
init_stop_date_int=$(date -d"${STOP_DATE}" +%Y%m%d)
temp_stop_date=$(date --date="${START_DATE} + ${DATE_DIFF} day" +%Y-%m-%d)
temp_stop_date=$(date --date="${temp_stop_date} - 1 day" +%Y-%m-%d)
temp_start_date=$START_DATE
temp_start_date_int=$(date -d"${temp_start_date}" +%Y%m%d)
temp_stop_date_int=$(date -d"${temp_stop_date}" +%Y%m%d)

# Specify the arXiv categories to harvest
sets="cs,econ,eess,math,physics,physics:astro-ph,physics:cond-mat,physics:gr-qc,physics:hep-ex,physics:hep-lat,physics:hep-ph,physics:hep-th,physics:math-ph,physics:nlin,physics:nucl-ex,physics:nucl-th,physics:physics,physics:quant-ph,q-bio,q-fin,stat"

while [ "$temp_stop_date_int" -le "$init_stop_date_int" -a "$temp_start_date_int" -le "$init_stop_date_int"  -a "$temp_start_date_int" -le "$temp_stop_date_int" ]
do
    temp_start_date=$(date -d"${temp_start_date}" +%Y-%m-%d)
    temp_stop_date=$(date -d"${temp_stop_date}" +%Y-%m-%d)
    echo "Harvesting from ${temp_start_date} to ${temp_stop_date}"
    docker-compose run --rm web inspirehep crawler schedule arXiv article --kwarg sets=$sets --kwarg from_date=$temp_start_date --kwarg until_date=$temp_stop_date
    temp_start_date=$(date -d"${temp_stop_date} + 1 day" +%Y-%m-%d)
    temp_start_date_int=$(date -d"${temp_start_date}" +%Y%m%d)
    temp_stop_date=$(date -d"${temp_stop_date} + ${DATE_DIFF} day" +%Y-%m-%d)
    temp_stop_date_int=$(date -d"${temp_stop_date}" +%Y%m%d)
    if [ "$temp_stop_date_int" -gt "$init_stop_date_int" -a "$temp_start_date_int" -le "$init_stop_date_int" ];
    then
        echo "Harvesting from ${temp_start_date} to ${STOP_DATE}"
        docker-compose run --rm web inspirehep crawler schedule arXiv article --kwarg sets=$sets --kwarg from_date=$temp_start_date --kwarg until_date=$STOP_DATE
    fi
    # The command below fetches the crawler job list. Second, it finds the header which contains the terms 'id    job_id', and
    # it fetches all the line after that. Lastly, it selects only the top line, which corresponds to the most recent harvest job.
    job=$(docker-compose run --rm web inspirehep crawler job list | sed -e '1,/id    job_id/d' | head -1)
    # We repeatedly check whether the output of the above command contains the term "None" (which can be found in the logs
    # and results columns of the crawler job list), since "None" corresponds to a harvest which is unfinished. Only when that
    # job finishes it exits the loop below and starts the next harvest.
    while [[ $job = *"None"* ]]
    do
        sleep 15
        job=$(docker-compose run --rm web inspirehep crawler job list | sed -e '1,/id    job_id/d' | head -1)
    done
done
