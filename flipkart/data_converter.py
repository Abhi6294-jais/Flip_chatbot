# import pandas as pd
# import os
# from langchain_core.documents import Document

# def dataconverter():

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     csv_path = os.path.join(BASE_DIR, "..", "data", "flipkart_product_review.csv")

#     product_data = pd.read_csv(csv_path)

#     data = product_data[["product_title", "review"]]

#     product_list = []

#     for index, row in data.iterrows():
#         item = {
#             "product_name": row["product_title"],
#             "review": row["review"]
#         }
#         product_list.append(item)

#     docs = []
#     for entry in product_list:
#         metadata = {"product_name": entry["product_name"]}
#         doc = Document(
#             page_content=entry["review"],
#             metadata=metadata
#         )
#         docs.append(doc)

#     return docs







import pandas as pd
import os
from langchain_core.documents import Document

def dataconverter():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "..", "data", "flipkart_product_review.csv")

    product_data = pd.read_csv(csv_path)

    # Keep only required columns
    data = product_data[["product_title", "review"]].dropna()

    docs = []

    for _, row in data.iterrows():
        doc = Document(
            page_content=str(row["review"]),
            metadata={"product_name": str(row["product_title"])}
        )
        docs.append(doc)

    return docs