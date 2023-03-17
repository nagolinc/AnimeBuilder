import os
import openai
import argparse
from flask import Flask, render_template, request, jsonify
import json
import animeCreator
from animeCreator import AnimeBuilder, getFilename
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
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = animeBuilder.create_characters(story_objects,novel_summary)
    return jsonify(characters)


@app.route('/create_chapters', methods=['POST'])
def create_chapters():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    num_chapters = int(data.get("numChapters", 3))
    chapters = animeBuilder.create_chapters(
        story_objects,novel_summary, characters, num_chapters)
    return jsonify(chapters)


@app.route('/create_scenes', methods=['POST'])
def create_scenes():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    chapters = data.get("chapters", [])
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))
    all_scenes = animeBuilder.create_scenes(
        story_objects,novel_summary, characters, chapters, num_chapters, num_scenes)
    return jsonify(all_scenes)


movies = {}


@app.route('/create_movie', methods=['POST'])
def create_movie():
    data = request.get_json()
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get('novelSummary')
    characters = data.get('characters')
    chapters = data.get('chapters')
    all_scenes = data.get('scenes')
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))

    movie_id = getFilename("", "mov")
    # movie_generator = animeBuilder.generate_movie_data(novel_summary, characters, chapters, scenes)
    movie_generator = animeBuilder.generate_movie_data(
        story_objects,novel_summary, characters, chapters, all_scenes, num_chapters, num_scenes)
    movies[movie_id] = MovieGeneratorWrapper(movie_generator)

    # return jsonify({"movie_id": movie_id})
    return jsonify(movie_id)


import concurrent.futures

class MovieGeneratorWrapper:
    def __init__(self, generator):
        self.generator = generator
        self._next_element_future = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._fetch_next_element()

    def _get_next_element(self):
        try:
            element = next(self.generator)
            if "image" in element:
                fName =getFilename(savePath, "png")
                element["image"].save( fName)
                element["image"] = fName
            return element
        except StopIteration:
            return None

    def _fetch_next_element(self):
        self._next_element_future = self._executor.submit(self._get_next_element)

    def get_next_element(self):
        current_element = self._next_element_future.result()
        if current_element is not None:
            self._fetch_next_element()
        return current_element


@app.route('/get_next_element/<string:movie_id>', methods=['GET'])
def get_next_element(movie_id):
    movie_generator = movies.get(movie_id)

    if movie_generator is None:
        return jsonify({"error": "Movie not found"}), 404

    element = movie_generator.get_next_element()
    if element is None:
        return jsonify({"done": "No more elements"}), 200

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
    

    parser.add_argument('--negativePrompt', type=str,
                        default="collage, grayscale, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
                        help="negative prompt")
    
    parser.add_argument('--extraTemplatesFile', type=str,
                        default=None,
                        help="file with template overrides")

    args = parser.parse_args()

    animeBuilder = AnimeBuilder(num_inference_steps=15,
                                textModel="GPT3",
                                diffusionModel=args.modelName,
                                doImg2Img=True,
                                negativePrompt=args.negativePrompt,
                                suffix=args.promptSuffix
                                )
    
    if args.extraTemplatesFile:
        with open(args.extraTemplatesFile, "r") as file:
            code = file.read()
            templateOverrides=eval(code)
            for k,v in templateOverrides.items():
                animeBuilder.templates[k]=v
    



    app.run(debug=True)
