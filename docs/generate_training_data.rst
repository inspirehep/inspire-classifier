..
    This file is part of INSPIRE.
    Copyright (C) 2014-2018 CERN.

    INSPIRE is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    INSPIRE is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.

    In applying this license, CERN does not waive the privileges and immunities
    granted to it by virtue of its status as an Intergovernmental Organization
    or submit itself to any jurisdiction.

==========================
Generate the training data
==========================

We need annotated data to train the classifier in a supervised way. The data should contain title and abstract as well as their corresponding labels i.e. Rejected, Non-Core, or Core. We do have that data available in INSPIRE due to the large number of daily harvests and the corresponding curator actions to classify them in one of the aforementioned categories.

The training data needs to be generated and organized in an appropriate format. The process, however, is somewhat complex. Data for Core and Non-Core records needs to be generated separately from data for the Rejected records.

Generate Core and Non-Core data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core and Non-Core records can be extracted from the INSPIRE dump data migrated on a local inspire-next instance. First, we would need to `setup inspire-next locally <https://inspirehep.readthedocs.io/en/latest/getting_started.html>`_. Next, we need to migrate the data from a recent `INSPIRE dump <http://inspirehep.net/dumps/inspire-dump.html>`_. The HEP dump is the relevant one in this case. Download that, and from the local inspire-next directory, go to the inspirehep terminal:

::

    docker-compose run --rm web bash

In the terminal, you can migrate from the downloaded dump using:

::

    APP_DEBUG=False inspirehep migrate file -w <path_to_downloaded_file>


Then, the content of code snippet specified in ``generate_core_and_noncore_data.py`` is to be copied and executed from within the inspirehep shell [1]_.

This produces two files: *inspire_core_records.json* and *inspire_noncore_records.json*. We will later combine this data with the Rejected records data.

Here, we consider only data from 2016 onwards since before that the curation rules for classification as Core, Non-Core, and Rejected were different. However, the user is free to modify the STARTING_DATE variable in the script to specify a starting date of their choice.

Generate Rejected data
^^^^^^^^^^^^^^^^^^^^^^

The data for Rejected articles is harvested from the local inspire-next instance in a hackish way. The workflows themselves need to be modified in our local inspire-next setup. First, the file *inspire-next/inspirehep/modules/workflows/workflows/article.py* needs to be modified as specified in ``article.py``. We need to add another file *inspire-next/inspirehep/modules/workflows/tasks/makejson.py* with the contents of ``makejson.py``.

Once the workflow has been modified, we are ready to start the harvest. First, we need to deploy the harvest spiders. This can be done from the *inspire-next* instance folder:

::

    docker-compose -f docker-compose.deps.yml run --rm scrapyd-deploy

To trigger the harvest, we need to copy the file ``trigger_harvest.sh`` to the local inspire-next directory and then run it as:

::

    chmod +x trigger_harvest.sh
    ./trigger_harvest.sh -s <starting_date> -d <date_diff> -e <last_date>

For example:

::

    ./trigger_harvest.sh -s 2018-10-02 -d 2 -e 2018-10-10

This would trigger a harvest from 2nd October 2018 to 10th October 2018, both days included, with 2 days of records harvested at a time. If no arguments are specified, the script would only trigger harvests for today and yesterday.

The core command in *trigger_harvest.sh* is as follows, which schedules a harvest from arXiv from the specified *from_date* to the specified *until_date* over a set of arXiv categories defined in the *sets* argument.

::

    docker-compose run --rm web inspirehep crawler schedule arXiv article --kwarg 'from_date=2018-11-06' --kwarg 'until_date=2018-11-07' --kwarg 'sets=cs,econ,eess,math,physics,physics:astro-ph,physics:cond-mat,physics:gr-qc,physics:hep-ex,physics:hep-lat,physics:hep-ph,physics:hep-th,physics:math-ph,physics:nlin,physics:nucl-ex,physics:nucl-th,physics:physics,physics:quant-ph,q-bio,q-fin,stat'

**Important**: Make sure that the dates (especially the <last_date> which is also the until_date) correspond to the date of the INSPIRE dump used for record migration. This is important as later on, we also need to generate the list of Core and Non-Core records from the same INSPIRE dump and any records harvested from after that date will be considered Rejected and that will corrupt our dataset.

Since the harvest will generate a large number of files, it's necessary to delete files associated with records from the workflows which have already finished. Otherwise, the harvest will fill our disk space quickly. We can use any job or task scheduler for this as we need to delete the content created in *$DOCKER_DATA/tmp/virutalenv/var/data/workflows/files"* periodically, where *$DOCKER_DATA* is the environment variable corresponding to our *inspire-next* docker data. We can use *crontab* as the task scheduler in linux. In the bash terminal:

::

    crontab -e

This will open our favorite text editor (or we'll be required to set it). Add the following line at the end of the file and save and quit:

::

    */15 * * * * find $DOCKER_DATA/tmp/virtualenv/var/data/workflows/files/* -mmin +30 -delete

This will schedule a task to run every 15 minutes which will find and delete all files created before the last 30 minutes. It's recommended to schedule the cronjob after starting the harvests since the first harvests and workflows can take a few minutes to start. We can schedule the command to run more frequently or vice versa depending on our hardware specifications.

The harvest produces a file named *inspire_harvested_data.json*. We can monitor the harvest status in the local holdingpen. However, it doesn't contain information on whether the harvested records were Core, Non-Core, or Rejected. To find this, we need to extract the list of arXiv identifiers of Core and Non-Core records from our local inspire-next instance. From the *inspirehep shell* [1]_, copy the contents of ``get_core_and_noncore_arXiv_identifiers.py`` and execute. This will produce two files, *inspire_core_list.txt* and *inspire_noncore_list.txt*. These files will be used to filter out Core and Non-Core records from the harvested data.

Combine the Core, Non-Core, and Rejected data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Core, Non-Core, and Rejected data can be combined by using the python script found at ``combine_core_noncore_rejected_data.py``. The different required files paths need to be specified in the file before running the script. Finally, this will produce the file *inspire_data.df* which is a Pandas DataFrame and which can be used for training and evaluation of the INSPIRE classifier. This file should be placed at the path specified in *inspire-classifier/inspire_classifier/config.py* in the variable *CLASSIFIER_DATAFRAME_PATH*.

The resulting pandas dataframe will contain 2 columns: *labels* and *text* where *text* is *title* and *abstract* concatenated with a *<ENDTITLE>* token in between.



.. [1] The inspirehep shell can be accessed by accessing the *inspire-next* bash terminal:

    ::

        docker-compose run --rm web bash

    Once inside, the terminal, you can run the following to access the inspirehep shell/cli:

    ::

        inspirehep shell

