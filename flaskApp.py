import os
import openai
import argparse
from flask import Flask, render_template, request, jsonify
import json
import animeCreator
from animeCreator import AnimeBuilder
import uuid

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create_summary', methods=['POST'])
def create_summary():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = animeBuilder.create_novel_summary(story_objects)
    return jsonify(novel_summary)


@app.route('/create_characters', methods=['POST'])
def create_characters():
    data = request.get_json()
    novel_summary = data.get("novelSummary", {})
    characters = animeBuilder.create_characters(novel_summary)
    return jsonify(characters)


@app.route('/create_chapters', methods=['POST'])
def create_chapters():
    data = request.get_json()
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    num_chapters = int(data.get("numChapters", 3))
    chapters = animeBuilder.create_chapters(
        novel_summary, characters, num_chapters)
    return jsonify(chapters)


@app.route('/create_scenes', methods=['POST'])
def create_scenes():
    data = request.get_json()
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    chapters = data.get("chapters", [])
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))
    all_scenes = animeBuilder.create_scenes(
        novel_summary, characters, chapters, num_chapters, num_scenes)
    return jsonify(all_scenes)


movies = {}


@app.route('/create_movie', methods=['POST'])
def create_movie():
    data = request.get_json()
    novel_summary = data.get('novelSummary')
    characters = data.get('characters')
    chapters = data.get('chapters')
    all_scenes = data.get('scenes')
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))

    movie_id = str(uuid.uuid4())
    # movie_generator = animeBuilder.generate_movie_data(novel_summary, characters, chapters, scenes)
    movie_generator = animeBuilder.generate_movie_data(
        novel_summary, characters, chapters, all_scenes, num_chapters, num_scenes)
    movies[movie_id] = movie_generator

    # return jsonify({"movie_id": movie_id})
    return jsonify(movie_id)


@app.route('/get_next_element/<string:movie_id>', methods=['GET'])
def get_next_element(movie_id):
    movie_generator = movies.get(movie_id)

    if movie_generator is None:
        return jsonify({"error": "Movie not found"}), 404

    try:
        element = next(movie_generator)
    except StopIteration:
        return jsonify({"error": "No more elements"}), 400

    if "image" in element:
        fName = str(uuid.uuid4())+".png"
        element["image"].save(savePath+fName)
        element["image"] = savePath+fName

    return jsonify(element)


openai.api_key = os.environ['OPENAI_API_KEY']

if __name__ == '__main__':

    savePath = "./static/samples/"

    parser = argparse.ArgumentParser(
        description="Flask App with model name parameter")
    parser.add_argument('--modelName', type=str,
                        default="andite/anything-v4.0", help="Name of the model")
    parser.add_argument('--promptSuffix', type=str,
                        default=", anime drawing", help="add to image prompt")
    args = parser.parse_args()

    animeBuilder = AnimeBuilder(num_inference_steps=15,
                                textModel="GPT3",
                                diffusionModel=args.modelName,
                                doImg2Img=True,
                                suffix=args.promptSuffix
                                )

    app.run(debug=False)
