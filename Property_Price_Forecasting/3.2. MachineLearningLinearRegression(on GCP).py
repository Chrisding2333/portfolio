# py file i used for GCP running
import time
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("Finals") \
    .getOrCreate()

# London
london_filtered = spark.read.csv("gs://dataproc-staging-us-central1-1035642532413-xtkeotnc/london.csv", header=True, inferSchema=True, timestampFormat="yyyy-MM-dd HH:mm:ss")
#Outside London
non_london_filtered = spark.read.csv("gs://dataproc-staging-us-central1-1035642532413-xtkeotnc/non london.csv", header=True, inferSchema=True, timestampFormat="yyyy-MM-dd HH:mm:ss")

# Convert features to index for one hot encoder for LONDON dataset
indexers_ln = [
    StringIndexer(inputCol="Property_Type", outputCol="Property_Type_index"),
    StringIndexer(inputCol="Old/New", outputCol="Old/New_index"),
    StringIndexer(inputCol="Duration", outputCol="Duration_index")
]

# One hot encoder from index to vectors for LONDON dataset
encoders_ln = [
    OneHotEncoder(inputCols=["Property_Type_index"], outputCols=["Property_Type_vec"]),
    OneHotEncoder(inputCols=["Old/New_index"], outputCols=["Old/New_vec"]),
    OneHotEncoder(inputCols=["Duration_index"], outputCols=["Duration_vec"])
]

# Convert features to index for one hot encoder for NON_LONDON dataset
indexers_nln = [
    StringIndexer(inputCol="Property_Type", outputCol="Property_Type_index"),
    StringIndexer(inputCol="Old/New", outputCol="Old/New_index"),
    StringIndexer(inputCol="Duration", outputCol="Duration_index"),
    StringIndexer(inputCol="Town/City", outputCol="Town/City_index")
]

# One hot encoder from index to vectors for NON_LONDON dataset
encoders_nln = [
    OneHotEncoder(inputCols=["Property_Type_index"], outputCols=["Property_Type_vec"]),
    OneHotEncoder(inputCols=["Old/New_index"], outputCols=["Old/New_vec"]),
    OneHotEncoder(inputCols=["Duration_index"], outputCols=["Duration_vec"]),
    OneHotEncoder(inputCols=["Town/City_index"], outputCols=["City_vec"])
]

# Combine indexers and encoders into a single pipeline
pipeline_ln = Pipeline(stages=indexers_ln + encoders_ln)
pipeline_nln = Pipeline(stages=indexers_nln + encoders_nln)

# Apply the pipeline to the DataFrame
new_london_df = pipeline_ln.fit(london_filtered).transform(london_filtered)
new_non_london_df = pipeline_nln.fit(non_london_filtered).transform(non_london_filtered)

df_V_ln = new_london_df
df_V = new_non_london_df

# Counter for location frequency
location_frequency = df_V.groupBy("Location").count()
location_frequency_ln = df_V_ln.groupBy("Location").count()

# Transformer for timestamp data
df_V = df_V.join(location_frequency, "Location")
df_V_ln = df_V_ln.join(location_frequency_ln, "Location")
df_V = df_V.withColumnRenamed("count", "location_frequency")
df_V_ln = df_V_ln.withColumnRenamed("count", "location_frequency")
df_V = df_V.withColumn("Year", F.year("Month")).withColumn("Month_Num", F.month("Month"))
df_V_ln = df_V_ln.withColumn("Year", F.year("Month")).withColumn("Month_Num", F.month("Month"))

# Drop non vectorized columns
df_V = df_V.drop("Transaction_unique_identifier","Location","Date_of_Transfer","Property_Type","Old/New","Duration","Month", "Property_Type_index", "Old/New_index","Duration_index", "Town/City", "Town/City_index")
df_V_ln = df_V_ln.drop("Transaction_unique_identifier","Location","Date_of_Transfer","Property_Type","Old/New","Duration","Month", "Property_Type_index", "Old/New_index","Duration_index","Town/City")

# Fetches festure col together
feature_cols = ["City_vec","location_frequency","Month_Num","Year","Property_Type_vec","Old/New_vec","Duration_vec"]
feature_cols_ln = ["location_frequency","Month_Num","Year","Property_Type_vec","Old/New_vec","Duration_vec"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler_ln = VectorAssembler(inputCols=feature_cols_ln, outputCol="features")

df_V = assembler.transform(df_V)
df_V_ln = assembler_ln.transform(df_V_ln)

# Split LONDON dataset
fftyp_ln = df_V_ln.randomSplit([0.5, 0.5])[0]
ttyfp_ln = fftyp_ln.randomSplit([0.5, 0.5])[0]
tenp_ln = ttyfp_ln.randomSplit([0.4, 0.6])[0]

# Tried 10%, 25%, 50%, 100% of data for cloud computing effciency comparsion
subsets_ln = {
    "10%": tenp_ln,
    "25%": ttyfp_ln,
    "50%": fftyp_ln,
    "100%": df_V_ln
}


train_test_splits = {size: subset.randomSplit([0.7, 0.3]) for size, subset in subsets_ln.items()}

train_10_ln, test_10_ln = train_test_splits["10%"]
train_25_ln, test_25_ln = train_test_splits["25%"]
train_50_ln, test_50_ln = train_test_splits["50%"]
train_100_ln, test_100_ln = train_test_splits["100%"]

# Split non LONDON dataset
fftyp = df_V.randomSplit([0.5, 0.5])[0]
ttyfp = fftyp.randomSplit([0.5, 0.5])[0]
tenp = ttyfp.randomSplit([0.4, 0.6])[0]

# Tried 10%, 25%, 50%, 100% of data for cloud computing effciency comparsion
subsets = {
    "10%": tenp,
    "25%": ttyfp,
    "50%": fftyp,
    "100%": df_V
}

train_test_splits = {size: subset.randomSplit([0.7, 0.3]) for size, subset in subsets.items()}

train_10, test_10 = train_test_splits["10%"]
train_25, test_25 = train_test_splits["25%"]
train_50, test_50 = train_test_splits["50%"]
train_100, test_100 = train_test_splits["100%"]

start_time = time.time()


def train_and_evaluate(train_df, test_df, param, dataset_name, loc_name):
    results = []

        # Initialize the LinearRegression model
    model = LinearRegression(featuresCol="features", labelCol='price', 
                                 elasticNetParam=param['elasticNetParam'], 
                                 regParam=param['regParam'], 
                                 maxIter=param['maxIter'])

        # Train the model
    trained_model = model.fit(train_df)

        # Make predictions on the test set
    predictions = trained_model.transform(test_df)

        # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)


    end_time = time.time()
    duration = end_time - start_time

        # Append the results
    results.append(Row(Location_name=loc_name, Dataset=dataset_name, Model="Linear Regression", RMSE=rmse, Duration=duration))

    return results

# Best parameters tested out
param_lr = {'elasticNetParam': 1.0, 'regParam': 0.1, 'maxIter': 100}

# Runs model on different dataset for cloud computing effciency comparsion
results_overall_lr = []
results_overall_lr.extend(train_and_evaluate(train_10_ln, test_10_ln, param_lr, "10%", "London"))
results_overall_lr.extend(train_and_evaluate(train_10, test_10,param_lr,"10%","Non_London"))

results_overall_lr.extend(train_and_evaluate(train_25_ln, test_25_ln, param_lr, "25%", "London"))
results_overall_lr.extend(train_and_evaluate(train_25, test_25,param_lr,"25%","Non_London"))

results_overall_lr.extend(train_and_evaluate(train_50_ln, test_50_ln, param_lr, "50%", "London"))
results_overall_lr.extend(train_and_evaluate(train_50, test_50,param_lr,"50%","Non_London"))

results_overall_lr.extend(train_and_evaluate(train_100_ln, test_100_ln, param_lr, "100%", "London"))
results_overall_lr.extend(train_and_evaluate(train_100, test_100,param_lr, "100%", "Non_London"))

print(results_overall_lr)

