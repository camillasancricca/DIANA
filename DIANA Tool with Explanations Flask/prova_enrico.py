from neo4j import GraphDatabase
from joblib import dump, load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prova_db():

    # prova db
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "ciaociao")

    driver = GraphDatabase.driver(URI, auth=AUTH)

    driver.verify_connectivity()
    print("verify connectivity tutto ok\n")

    """
    # Get the methods
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $name})-[:IMPLEMENTED_WITH]-(m) RETURN m",
        name="Outlier Detection",
        database_="neo4j",
    )

    # Loop through results and do something with them
    for method in records:
        print(method)

    # Summary information
    print("The query `{query}` returned {records_count} records in {time} ms.".format(
        query=summary.query, records_count=len(records),
        time=summary.result_available_after,
    ))
    
    """

    dimensions = ["Uniqueness", "Completeness", "Accuracy"]

    for dimension in dimensions:

        # Get the techniques that improve that dimension
        records, summary, keys = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[a:AFFECTS]->(d:DQ_DIMENSION) \
            WHERE a.influence_type = $influence_type and d.name = $dimension_name \
            RETURN n.name AS name",
            influence_type="Improvement",
            dimension_name=dimension,
            database_="neo4j",
        )
        for tech in records:
            print(tech["name"])
            # Here I query the methods for that technique
            records_m, summary_m, keys_m = driver.execute_query(
                "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
                WHERE n.name = $technique_name \
                RETURN m.name AS name",
                technique_name=tech["name"],
                database_="neo4j",
            )
            for meth in records_m:
                print(meth["name"])
                print({"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": dimension.upper()})

    records, summary, keys = driver.execute_query(
        "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:BENEFITS_FROM]-(ml:ML_APPLICATION) \
         WHERE ml.application_method = $ml_algorithm \
         RETURN DISTINCT n.name AS name",
        ml_algorithm="Logistic Regression",
        database_="neo4j"
    )
    for tech in records:
        print(tech["name"])
        # Here I query the methods for that technique
        records_m, summary_m, keys_m = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
            WHERE n.name = $technique_name \
            RETURN m.name AS name",
            technique_name=tech["name"],
            database_="neo4j",
        )
        for meth in records_m:
            print(meth["name"])
            print({"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": "ML_ORIENTED_ACTIONS"})


    driver.close()



def prova_classifier():
    trained_model = load('trained_classifier.joblib')

    dataset = pd.read_csv("dataset_classifier_features.csv")

    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])

    ml_columns = ["ML_ALGORITHM_dt", "ML_ALGORITHM_lr", "ML_ALGORITHM_knn", "ML_ALGORITHM_nb"]
    missing_cols = set(ml_columns) - set(dataset.columns)

    for c in missing_cols:
        dataset[c] = 0

    # feature_cols = list(dataset.columns)
    feature_cols = ['n_tuples', 'n_attributes', 'p_num_var', 'p_cat_var', 'p_duplicates',
                    'total_size', 'p_avg_distinct', 'p_max_distinct', 'p_min_distinct',
                    'avg_density', 'max_density', 'min_density', 'avg_entropy', 'max_entropy',
                    'min_entropy', 'p_correlated_features', 'max_pearson', 'min_pearson',
                    'ML_ALGORITHM_dt', 'ML_ALGORITHM_knn', 'ML_ALGORITHM_lr', 'ML_ALGORITHM_nb']

    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    # feature_cols.remove("name")

    dataset = dataset[0:][feature_cols]  # Features

    # faccio scaling

    scaler = load('trained_scaler.joblib')
    # scaler = StandardScaler()
    # scaler = RobustScaler()

    dataset = scaler.transform(dataset)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    best_technique_predicted = trained_model.predict(dataset)

    print(best_technique_predicted)
    print(best_technique_predicted[0])
    return best_technique_predicted


def prova_query_descrittiva():

    # prova db
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "ciaociao")

    driver = GraphDatabase.driver(URI, auth=AUTH)

    driver.verify_connectivity()
    print("verify connectivity tutto ok\n")

    # technique = "Imputation"
    method = "No Imputation"

    # I query what is the technique for that method

    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
         WHERE m.name = $method \
         RETURN t.name as technique",
        method=method,
        database_="neo4j",
    )

    technique = records[0]
    technique = technique["technique"]
    print(technique)

    # dimensions affected
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})-[a:AFFECTS]->(d:DQ_DIMENSION) \
         RETURN d.name AS dimension, a.influence_type AS influence_type ",
        technique=technique,
        database_="neo4j",
    )

    dimensions = []
    dimensions_influences = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        dimensions.append(record["dimension"])
        dimensions_influences.append(record["influence_type"])
    print(dimensions)
    print(dimensions_influences)

    # ml affected
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})<-[:BENEFITS_FROM]-(ml:ML_APPLICATION) \
         RETURN ml.application_method AS ml_algorithm",
        technique=technique,
        database_="neo4j",
    )

    ml_algorithms = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        ml_algorithms.append(record["ml_algorithm"])
    print(ml_algorithms)

    # depends on feature, technique
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})-[d:DEPENDS_ON]->(p:DATA_PROFILE_FEATURE) \
        RETURN p.name AS feature, d.description AS description ",
        technique=technique,
        database_="neo4j",
    )

    t_features = []
    t_features_descriptions = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        t_features.append(record["feature"])
        t_features_descriptions.append(record["description"])
    print(t_features)
    print(t_features_descriptions)


 # depends on feature, method
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_METHOD{name: $method})-[d:DEPENDS_ON]->(p:DATA_PROFILE_FEATURE) \
        RETURN p.name AS feature, d.description AS description",
        method=method,
        database_="neo4j",
    )

    m_features = []
    m_features_descriptions = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        m_features.append(record["feature"])
        m_features_descriptions.append(record["description"])
    print(m_features)
    print(m_features_descriptions)

    text = "TECHNIQUE " + technique + " :\n"

    for dimension, dimension_influence in zip(dimensions, dimensions_influences):
        text = text + "Affects the dimension " + dimension + " in this way: " + dimension_influence + "\n"

    for ml_algorithm in ml_algorithms:
        text = text + "The ML algorithm " + ml_algorithm + " benefits from this technique" + "\n"

    for t_feature, t_feature_description in zip(t_features, t_features_descriptions):
        text = text + "Depends on the dataset feature " + t_feature + " in this way: " + t_feature_description + "\n"

    text = text + "Method " + method + " is a possible implementation method for this technique" + "\n"

    for m_feature, m_feature_description in zip(m_features, m_features_descriptions):
        text = text + "This method depends on the dataset feature " + m_feature + " in this way: " + m_feature_description + "\n"

    print(text)

    driver.close()



if __name__ == "__main__":
    # prova_db()
    # prova_classifier()
    prova_query_descrittiva()






