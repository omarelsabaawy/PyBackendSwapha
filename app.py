from flask import Flask, jsonify
import pymongo
import pandas as pd
from model import find_and_save_matches
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from flask_cors import CORS

sbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

app = Flask(__name__)
CORS(app)

def get_user_list(user_id):
    # establish a connection to the MongoDB instance
    client = pymongo.MongoClient('mongodb+srv://swapha-app:YGzPJW1MV4AMNZ97@cluster0.h3wkjeo.mongodb.net/test')
    # select the database and collection to query
    db = client.test
    # get the user's Swap and Wish lists
    swap_list = list(db.SwapList.aggregate([{"$match": {"product.userId": user_id}}]))
    Swap = pd.json_normalize(swap_list)

    wish_list = list(db.WishList.aggregate([{"$match": {"userId": user_id}}]))
    Wish = pd.json_normalize(wish_list)

    # get the set of categories in the user's Swap and Wish lists
    swap_categories = set(Swap['product.category'].unique())
    wish_categories = set(Wish['category'].unique())

    # filter the SwapList collection by the user's Wish categories
    swap_list1 = list(db.SwapList.aggregate([
        {"$match": {"product.userId": {"$ne": user_id}, "product.category": {"$in": list(wish_categories)}}}
    ]))
    Swap1 = pd.json_normalize(swap_list1)

    # filter the WishList collection by the user's Swap categories
    wish_list1 = list(db.WishList.aggregate([
        {"$match": {"userId": {"$ne": user_id}, "category": {"$in": list(swap_categories)}}}
    ]))
    wish1 = pd.json_normalize(wish_list1)
    Swap = Swap.rename(columns={'product._id': '_id', 'product.name': 'name', 'product.category': 'category',
                                'product.userId': 'userId', 'product.owner': 'owner', 'product.imageUrl': 'imageUrl','product.desc': 'desc'})

    Swap1 = Swap1.rename(columns={'product._id': '_id', 'product.name': 'name', 'product.category': 'category',
                                  'product.userId': 'userId', 'product.owner': 'owner', 'product.imageUrl': 'imageUrl','product.desc': 'desc'})
    return Swap, Swap1, Wish, wish1


@app.route("/")
def getHome():
    return "Swapha Engine Server is now Running!"


@app.route("/predict/products/<user_id>")
def predict(user_id):
    swap_user, swap_list, wish_user, wish_list = get_user_list(ObjectId(user_id))
    threshold = 0.7
    results = find_and_save_matches(sbert_model, swap_user, wish_user, swap_list, wish_list, threshold)
    print(results)
    if results:
       return jsonify({'products': results, 'status': True})
    else:
        return jsonify({'status': False, 'products': None})

if __name__ == "__main__":
    app.run(debug=True)
