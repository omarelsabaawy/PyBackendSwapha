from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from bson import ObjectId
import json

def find_and_save_matches(sbert_model, swap_user, wish_user, swap_list, wish_list, threshold):
    # extract categories of items that user X has in swap_user
    user_x_swap_categories = set(swap_user['category'].unique())

    # extract categories of items that user X needs in wish_user
    user_x_wish_categories = set(wish_user['category'].unique())

    # filter swap_list and wish_list to only include items with matching categories
    relevant_swap_items = swap_list[swap_list['category'].isin(user_x_wish_categories) & (swap_list['category'] == wish_user['category'].iloc[0])]
    relevant_wish_items = wish_list[wish_list['category'].isin(user_x_swap_categories) & (wish_list['category'] == swap_user['category'].iloc[0])]

    # extract the descriptions of the relevant items
    swap_desc = relevant_swap_items['desc'].tolist()
    wish_desc = relevant_wish_items['desc'].tolist()

    # embed the descriptions using the sbert model
    swap_emb = sbert_model.encode(swap_desc)
    wish_emb = sbert_model.encode(wish_desc)

    # calculate the cosine similarity matrix between user X's wish items and other users' swap items
    similarity_matrix = cosine_similarity(swap_emb, wish_emb)

    # create a data frame with the results
    results = pd.DataFrame({
        'user_id': relevant_swap_items['userId'].tolist(),
        'item_id': relevant_swap_items['_id'].tolist(),
        'item_name': relevant_swap_items['name'].tolist(),
        'desc': relevant_swap_items['desc'].tolist(),
        'imageUrl': relevant_swap_items['imageUrl'].tolist(),
        'owner': relevant_swap_items['owner'].tolist(),
        'similarity': similarity_matrix.max(axis=1),
        'matched_wish_item': relevant_wish_items.iloc[similarity_matrix.argmax(axis=1)]['name'].tolist()
    })

    # filter the results based on the threshold
    results = results[results['similarity'] >= threshold]

    # sort the data frame by similarity in descending order
    results = results.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # convert ObjectId objects to strings
    for col in results.columns:
        if isinstance(results[col][0], ObjectId):
            results[col] = results[col].apply(lambda x: str(x))

    json_str = results.to_json(orient='records', indent=4)
    return json.loads(json_str)

