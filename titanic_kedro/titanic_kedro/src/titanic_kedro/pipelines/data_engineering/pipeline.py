from kedro.pipeline import node, Pipeline
from titanic_kedro.pipelines.data_engineering.nodes import (
    merge_train_test,
    clean_cabin,
    get_titles)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=merge_train_test,
                inputs=["train_raw","test_raw"],
                outputs="train_test_raw",
                name="train_test_raw",
            ),
            node(
                func=clean_cabin,
                inputs="train_test_raw",
                outputs="train_test_intermediate",
                name="train_test_intermediate",
            ),
            node(
                func=get_titles,
                inputs="train_test_intermediate",
                outputs="train_test_clean",
                name="train_test_clean",
            ),
        ]
          )
