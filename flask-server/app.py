from flask import Flask, render_template, request, redirect, url_for, session
import jsonlines
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
import dotenv

import sys

sys.path.append("..")
from game import RepeatedReferenceGame, Trial
from agents.openai_api_agent import OpenAIAPISpeaker
from agents.prompts import (
    SPEAKER_SYSTEM_PROMPT_EXPLICIT,
    SPEAKER_SYSTEM_PROMPT_STANDARD,
)

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a random secret key

# Configuration
STATIC_FOLDER = "static"
IMAGES_KEY = "coco-images"
IMAGES_DIR = os.path.join(STATIC_FOLDER, IMAGES_KEY)
RESULTS_FILE = "trials.jsonl"
TASKS_FILE = os.environ.get("TASKS_PATH", "tasks.jsonl")
SPEAKER = OpenAIAPISpeaker(
    "gpt-4.1",
    None,
    os.environ.get("OPENAI_API_KEY"),
    "static/coco-images",
    "once",
    True,
    {"max_completion_tokens": 128},
    system_prompt_template=(
        SPEAKER_SYSTEM_PROMPT_EXPLICIT
        if os.environ.get("PROMPT_TYPE", "explicit")
        else SPEAKER_SYSTEM_PROMPT_STANDARD
    ),
)


# Load all trials from the JSONL file
def load_trials():
    trials = []
    if os.path.exists(RESULTS_FILE):
        with jsonlines.open(RESULTS_FILE, "r") as reader:
            trials = list(reader)
    return trials


# Get a specific trial by ID
def get_trial_by_id(trial_id):
    trials = load_trials()
    for trial in trials:
        if trial.get("trial_id") == trial_id:
            return trial
    return None


def load_tasks():
    tasks = []
    if os.path.exists(TASKS_FILE):
        with jsonlines.open(TASKS_FILE, "r") as reader:
            tasks = list(reader)
            print(f"Loaded {len(tasks)} tasks from {TASKS_FILE}")
    return tasks


def get_task_by_id(task_id):
    tasks = load_tasks()
    for task in tasks:
        if task.get("task_id") == task_id:
            return task
    return None


# Reconstruct the chain of trials starting from a specific trial
def get_trial_chain(trial_id):
    chain = []
    current_id = trial_id

    while current_id:
        trial = get_trial_by_id(current_id)
        if not trial:
            break

        chain.append(trial)
        current_id = trial.get("parent_trial")

    # Reverse to get chronological order (oldest first)
    chain.reverse()
    return chain


# Generate dummy descriptions (in a real app, these would be meaningful)
DUMMY_DESCRIPTIONS = [
    "This is the target image you're looking for.",
    "Find the image that looks like this description.",
    "Can you identify the image based on this text?",
    "Select the image that best matches this description.",
]


@app.route("/")
def index():
    """Home page with a button to start a new trial"""
    return render_template("index.html")


@app.route("/back")
def go_back():
    """Go back one step in the trial chain and start a new trial from there"""
    current_parent_id = session.get("parent_trial_id")

    if not current_parent_id:
        # If there's no parent, just start a new trial
        return redirect(url_for("trial"))

    # Get the parent of the parent (grandparent)
    parent_trial = get_trial_by_id(current_parent_id)
    grandparent_id = parent_trial.get("parent_trial") if parent_trial else None

    # Start a new trial using the parent as the context source
    # but the grandparent as the parent_id (effectively branching)
    return redirect(
        url_for("trial", parent_id=grandparent_id, context_source_id=current_parent_id)
    )  # Load contexts from JSON file


@app.route("/trial")
def trial():
    """Generate a new trial and render it"""
    parent_trial_id = request.args.get("parent_id")
    # This allows us to use the context from one trial but set a different trial as parent
    context_source_id = request.args.get("context_source_id") or parent_trial_id

    # Check if this is a continuation of a previous trial
    if context_source_id:
        # Get the trial to use for context
        source_trial = get_trial_by_id(context_source_id)
        if source_trial:
            task = get_task_by_id(source_trial["task_id"])
        else:
            # Fallback to a new random context if source not found
            tasks = load_tasks()
            task = random.choice(tasks)
    else:
        # Start a new chain with a random context
        tasks = load_tasks()
        task = random.choice(tasks)

    # Generate unique ID for this trial
    trial_id = str(uuid.uuid4())

    # Store information for later verification
    session["trial_id"] = trial_id
    session["parent_trial_id"] = parent_trial_id
    session["task_id"] = task["task_id"]

    # Convert image paths to be relative to the static folder
    image_paths = [os.path.join(IMAGES_KEY, img) for img in task["context"]]

    # If this is a continued trial, get the previous trials chain
    previous_trials = []
    if parent_trial_id:
        previous_trials = get_trial_chain(parent_trial_id)

    trial_index = len(previous_trials)
    session["trial_index"] = trial_index

    game = RepeatedReferenceGame(
        context=task["context"],
        trials=[
            *[
                Trial(
                    target=task["targets"][i],
                    message=t["message"],
                    selection=t["selection"],
                    correct=t["correct"],
                )
                for i, t in enumerate(previous_trials)
            ],
            Trial(target=task["targets"][trial_index]),
        ],
    )
    session["description"] = SPEAKER.generate(game)[0]

    return render_template(
        "continuous_trial.html",
        images=image_paths,
        description=session["description"],
        previous_trials=previous_trials,
        trial_count=len(previous_trials) + 1,
        feedback=None,  # No feedback for the first trial
        labels=[
            chr(ord("A") + i) for i in range(len(task["context"]))
        ],  # Labels for the images
    )


@app.route("/submit", methods=["POST"])
def submit():
    """Handle the selection submission and start the next trial with the same context"""
    selection = Path(request.form.get("selected_image", None)).name

    # Retrieve trial information from session
    trial_id = session.get("trial_id")
    parent_trial_id = session.get("parent_trial_id")
    task_id = session.get("task_id")
    trial_index = session.get("trial_index")
    description = session.get("description", "")

    task = get_task_by_id(task_id)

    # Check if the selection matches the target
    is_correct = selection == task["targets"][trial_index]

    # Save trial data to JSONL file
    trial_data = {
        "trial_id": trial_id,
        "parent_trial": parent_trial_id,
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "trial_index": trial_index,
        "message": description,
        "selection": selection,
        "correct": is_correct,
    }

    # Append to JSONL file
    with jsonlines.open(RESULTS_FILE, mode="a") as writer:
        writer.write(trial_data)

    # Generate a new trial with the current one as parent
    new_trial_id = str(uuid.uuid4())

    # Store information for the new trial
    session["trial_id"] = new_trial_id
    session["parent_trial_id"] = trial_id  # Current trial becomes the parent
    session["task_id"] = task_id  # Keep the same context

    # Get the chain of trials including the one just completed
    previous_trials = get_trial_chain(trial_id) if trial_id else []
    # We need to add the current trial manually since it was just saved
    current_trial_already_in_chain = False
    for trial in previous_trials:
        if trial.get("trial_id") == trial_id:
            current_trial_already_in_chain = True
            break

    if not current_trial_already_in_chain:
        previous_trials.append(trial_data)

    session["trial_index"] = len(previous_trials)

    if session["trial_index"] >= len(task["targets"]):
        # If all trials are completed, redirect to the end page
        return redirect(url_for("index"))

    game = RepeatedReferenceGame(
        context=task["context"],
        trials=[
            *[
                Trial(
                    target=task["targets"][i],
                    message=t["message"],
                    selection=t["selection"],
                    correct=t["correct"],
                )
                for i, t in enumerate(previous_trials)
            ],
            Trial(target=task["targets"][session["trial_index"]]),
        ],
    )

    session["description"] = SPEAKER.generate(game)[0]

    # Convert image paths to be relative to the static folder
    image_paths = [os.path.join(IMAGES_KEY, img) for img in task["context"]]

    # Calculate statistics
    correct_count = sum(1 for t in previous_trials if t.get("correct", False))
    accuracy = (correct_count / len(previous_trials) * 100) if previous_trials else 0

    # Render the trial template with feedback from the previous trial
    return render_template(
        "continuous_trial.html",
        images=image_paths,
        description=session["description"],
        previous_trials=[
            {
                **t,
                "selection": chr(ord("A") + task["context"].index(t["selection"])),
                "target": chr(
                    ord("A") + task["context"].index(task["targets"][t["trial_index"]])
                ),
            }
            for t in previous_trials
        ],
        trial_count=session["trial_index"],
        feedback={
            "is_correct": is_correct,
            "selected": chr(ord("A") + task["context"].index(selection)),
            "target": chr(
                ord("A")
                + task["context"].index(task["targets"][session["trial_index"] - 1])
            ),
            "accuracy": round(accuracy, 1),
        },
        labels=[
            chr(ord("A") + i) for i in range(len(task["context"]))
        ],  # Labels for the images
    )


def main():
    # Ensure the required directories exist
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Create results file if it doesn't exist
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            pass  # Just create an empty file
        print(f"Created empty results file: {RESULTS_FILE}")

    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()
