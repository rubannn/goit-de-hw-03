from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_config import create_spark_session

DATA_DIR = "data"


def load_csv(spark: SparkSession, file_name: str) -> DataFrame:
    """Завантажує CSV-файл з папки data та повертає DataFrame."""
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{DATA_DIR}/{file_name}")
    )


def clean_data(df: DataFrame) -> DataFrame:
    """Очищує DataFrame, видаляючи рядки з пропущеними значеннями."""
    return df.dropna()


def calculate_total_purchases_by_category(
    purchases: DataFrame, products: DataFrame
) -> DataFrame:
    """Визначає загальну суму покупок за кожною категорією продуктів."""
    return (
        purchases
        .join(products, on="product_id")
        .withColumn("total", F.col("quantity") * F.col("price"))
        .groupBy("category")
        .agg(F.round(F.sum("total"), 2).alias("total_sum"))
    )


def calculate_total_purchases_by_category_for_age_group(
    purchases: DataFrame, products: DataFrame, users: DataFrame,
    age_from: int = 18, age_to: int = 25,
) -> DataFrame:
    """Визначає суму покупок за кожною категорією продуктів
    для вікової категорії від age_from до age_to включно.
    """
    filtered_users = users.filter(
        (F.col("age") >= age_from) & (F.col("age") <= age_to)
    )
    return (
        purchases
        .join(filtered_users, on="user_id")
        .join(products, on="product_id")
        .withColumn("total", F.col("quantity") * F.col("price"))
        .groupBy("category")
        .agg(F.round(F.sum("total"), 2).alias("total_sum"))
    )


def calculate_category_share_for_age_group(
    category_sums: DataFrame,
) -> DataFrame:
    """Визначає частку покупок за кожною категорією товарів
    від сумарних витрат вікової групи. Відсоток округлюється
    до другого знака після коми.
    """
    total_sum = category_sums.agg(F.sum("total_sum")).first()[0]
    return (
        category_sums
        .withColumn(
            "percentage",
            F.round(F.col("total_sum") / F.lit(total_sum) * 100, 2),
        )
        .orderBy(F.col("percentage").desc())
    )


def select_top_categories(shares: DataFrame, top_n: int = 3) -> DataFrame:
    """Вибирає top_n категорій продуктів з найвищим відсотком
    витрат споживачами заданої вікової групи.
    """
    return shares.orderBy(F.col("percentage").desc()).limit(top_n)


def build_cleaning_report(spark: SparkSession, counts: list[tuple[str, int, int]]) -> DataFrame:
    """Будує DataFrame-звіт з кількістю рядків до та після очищення
    для кожного вхідного файлу.
    """
    rows = [
        (file_name, rows_before, rows_after, rows_before - rows_after)
        for file_name, rows_before, rows_after in counts
    ]
    return spark.createDataFrame(
        rows, ["file", "rows_before", "rows_after", "rows_removed"]
    )


def main() -> None:
    spark = create_spark_session()

    raw_users = load_csv(spark, "users.csv")
    raw_purchases = load_csv(spark, "purchases.csv")
    raw_products = load_csv(spark, "products.csv")

    users = clean_data(raw_users)
    purchases = clean_data(raw_purchases)
    products = clean_data(raw_products)

    print("=== Завдання 2: Кількість рядків до та після очищення ===")
    cleaning_report = build_cleaning_report(
        spark,
        [
            ("users.csv", raw_users.count(), users.count()),
            ("purchases.csv", raw_purchases.count(), purchases.count()),
            ("products.csv", raw_products.count(), products.count()),
        ],
    )
    cleaning_report.show()

    print("=== Завдання 3: Загальна сума покупок за категоріями ===")
    total_by_category = calculate_total_purchases_by_category(purchases, products)
    total_by_category.orderBy(F.col("total_sum").desc()).show()

    print("=== Завдання 4: Сума покупок за категоріями для вікової групи 18-25 ===")
    total_by_category_18_25 = calculate_total_purchases_by_category_for_age_group(
        purchases, products, users, 18, 25
    )
    total_by_category_18_25.orderBy(F.col("total_sum").desc()).show()

    print("=== Завдання 5: Частка покупок за категоріями для вікової групи 18-25 ===")
    shares_18_25 = calculate_category_share_for_age_group(total_by_category_18_25)
    shares_18_25.show()

    print("=== Завдання 6: Топ-3 категорії за відсотком витрат для вікової групи 18-25 ===")
    top_categories = select_top_categories(shares_18_25, 3)
    top_categories.show()

    spark.stop()


if __name__ == "__main__":
    main()
