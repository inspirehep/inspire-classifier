import os

import click
import pandas as pd
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.connections import connections
from inspire_utils.record import get_value
from tqdm import tqdm

DECISIONS_MAPPING = {
    "core": {
        "index": "records-hep",
        "filter_query": Q("term", core=True),
        "label": 2,
    },
    "non_core": {
        "index": "records-hep",
        "filter_query": Q("term", core=False) | ~Q("exists", field="core"),
        "label": 1,
    },
    "rejected": {
        "index": "holdingpen-hep",
        "filter_query": Q("term", _extra_data__approved=False),
        "label": 0,
    },
}


class LiteratureSearch(Search):
    connection_holdingpen = connections.create_connection(
        hosts=["https://os-inspire-legacy-os1.cern.ch/es"],
        timeout=30,
        http_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify_certs=False,
        use_ssl=True,
    )
    connection_inspirehep = connections.create_connection(
        hosts=["https://os-inspire-prod.cern.ch/es"],
        timeout=30,
        http_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify_certs=False,
        use_ssl=True,
    )

    def __init__(self, index, **kwargs):
        if index == "holdingpen-hep":
            connection = LiteratureSearch.connection_holdingpen
        else:
            connection = LiteratureSearch.connection_inspirehep
        super().__init__(
            using=kwargs.get("using", connection),
            index=index,
        )


class InspireClassifierSearch(object):
    def __init__(self, index, query_filters, year_from, year_to, month_from, month_to):
        self.search = LiteratureSearch(index=index)
        self.year_from = year_from
        self.month_from = month_from
        self.year_to = year_to
        self.month_to = month_to

        # Training, validation and test data

        if index == "holdingpen-hep":
            self.source_fields = [
                "id",
                "metadata.abstracts",
                "metadata.titles",
                "metadata.inspire_categories",
            ]
            self.id_field = "id"
            self.title_field = "metadata.titles[0].title"
            self.abstract_field = "metadata.abstracts[0].value"
            self.inspire_categories_field = "metadata.inspire_categories.term"
            self.query_filters = [
                query_filters
                & Q(
                    "range",
                    metadata__acquisition_source__datetime={
                        "gte": f"{self.year_from}-{self.month_from}",
                        "lt": f"{self.year_to}-{self.month_to}",
                    },
                ),
            ]
        else:
            self.source_fields = ["id", "abstracts", "titles", "inspire_categories"]
            self.id_field = "id"
            self.title_field = "titles[0].title"
            self.abstract_field = "abstracts[0].value"
            self.inspire_categories_field = "inspire_categories.term"
            self.query_filters = [
                query_filters
                & Q("range", _created={"gte": f"{self.year_from}-{self.month_from}",
                                       "lt": f"{self.year_to}-{self.month_to}",}),
            ]

    def _postprocess_record_data(self, record_data):
        id = get_value(record_data, self.id_field)
        title = get_value(record_data, self.title_field)
        abstract = get_value(record_data, self.abstract_field)
        inspire_categories = get_value(record_data, self.inspire_categories_field, [])
        return {
            "id": int(id),
            "title": title,
            "abstract": abstract,
            "inspire_categories": inspire_categories,
        }

    def get_decision_query(self):
        query = self.search.query(
            "bool",
            filter=self.query_filters,
        ).params(size=9999, _source=self.source_fields)
        return query


def get_data_for_decisions(year_from, year_to, month_from, month_to):
    for decision in DECISIONS_MAPPING:
        inspire_search = InspireClassifierSearch(
            index=DECISIONS_MAPPING[decision]["index"],
            query_filters=DECISIONS_MAPPING[decision]["filter_query"],
            year_from=year_from,
            year_to=year_to,
            month_from=month_from,
            month_to=month_to,
        )
        query = inspire_search.get_decision_query()
        for record_es_data in tqdm(query.scan()):
            record_classifier_data = inspire_search._postprocess_record_data(
                record_es_data.to_dict()
            )
            record_classifier_data["label"] = DECISIONS_MAPPING[decision]["label"]
            yield record_classifier_data


def prepare_inspire_classifier_dataset(data, save_data_path):
    inspire_data_df = pd.DataFrame(data)
    inspire_data_df = inspire_data_df.drop(
        inspire_data_df[inspire_data_df.abstract.isna()].index
    )
    inspire_data_df["text"] = (
        inspire_data_df["title"] + " <ENDTITLE> " + inspire_data_df["abstract"]
    )
    inspire_classifier_data_df = inspire_data_df[["id", "inspire_categories", "label", "text"]]
    inspire_classifier_data_df.to_pickle(save_data_path)


@click.command()
@click.option("--year-from", type=int, required=True)
@click.option("--month-from", type=int, required=False, default=1)
@click.option("--year-to", type=int, required=True)
@click.option("--month-to", type=int, required=False, default=12)
def get_inspire_classifier_dataset(year_from, year_to, month_from, month_to):
    if year_to < year_from:
        raise ValueError("year_to must be before year_from")
    if month_to < month_from:
        raise ValueError("month_to must be before month_from")
    if month_to > 12 or month_from > 12 or month_to < 1 or month_from < 1:
        raise ValueError("month_to and month_from must be between 1 and 12")
    month_from = f"{month_from:02d}-01"
    month_to = f"{month_to:02d}-31"
    print(f"Fetching {year_from}-{month_from} to {year_to}-{month_to}")
    inspire_classifier_dataset_path = os.path.join(
        os.getcwd(), f"inspire_classifier_dataset_{year_from}-{month_from}_{year_to}-{month_to}.pkl"
    )
    data = get_data_for_decisions(year_from, year_to, month_from, month_to)
    prepare_inspire_classifier_dataset(data, inspire_classifier_dataset_path)


if __name__ == "__main__":
    get_inspire_classifier_dataset()
