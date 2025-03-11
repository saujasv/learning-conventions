import ipywidgets as widgets
from IPython.display import display, clear_output, Image
from typing import List, Tuple, Optional, Dict, Callable
import random
import io
import os
import sys
import json
from PIL import Image as PILImage
from collections import defaultdict

sys.path.append("..")

from game import RepeatedReferenceGame, Trial
from agents import GenerateSpeaker


class GameManager:
    """
    Manages the state and communication in the repeated reference game.
    Handles message generation and user selections.
    """

    def __init__(
        self,
        agent: GenerateSpeaker,
        game: "RepeatedReferenceGame",
        game_save_path: str,
        max_trials=None,
        next_target_policy="block",
        feedback_label: bool = True,
    ):
        self.agent = agent
        self.game = game
        self.current_trial_idx = 0
        self.game_save_path = game_save_path
        self.max_trials = max_trials  # Maximum number of trials (None for unlimited)
        self.next_target_policy = next_target_policy
        self.feedback_label = feedback_label

        # Callbacks for UI updates
        self.on_message_generated: Optional[Callable[[str], None]] = None
        self.on_selection_made: Optional[Callable[[str, bool], None]] = None
        self.on_trial_completed: Optional[Callable[[], None]] = None
        self.on_game_completed: Optional[Callable[[], None]] = None
        self.on_trial_created: Optional[Callable[[Trial], None]] = None

    def get_current_trial(self):
        """Get the current trial or None if completed"""
        # Ensure we have a current trial
        self._ensure_current_trial_exists()

        if self.current_trial_idx < len(self.game.trials):
            return self.game.trials[self.current_trial_idx]
        return None

    def is_game_completed(self):
        """Check if the game is completed"""
        if self.max_trials is None:
            # No max trials set, so the game is never completed
            return False
        return self.current_trial_idx >= self.max_trials

    def get_next_target(self):
        if self.next_target_policy == "random":
            return random.choice(self.game.context)
        elif self.next_target_policy == "repair":
            if self.current_trial_idx == 0:
                return random.choice(self.game.context)
            elif self.game.trials[self.current_trial_idx - 1].correct:
                return random.choice(self.game.context)
            else:
                return self.game.trials[self.current_trial_idx - 1].target
        elif self.next_target_policy == "block":
            current_block = self.game.trials[
                (self.current_trial_idx // len(self.game.context))
                * len(self.game.context) :
            ]
            targets_in_block = [trial.target for trial in current_block]
            targets_not_in_block = [
                target for target in self.game.context if target not in targets_in_block
            ]
            return random.choice(targets_not_in_block)

    def _ensure_current_trial_exists(self):
        """Ensure that a trial exists for the current index"""
        if self.current_trial_idx >= len(self.game.trials):
            # We need to create a new trial
            if (
                self.max_trials is not None
                and self.current_trial_idx >= self.max_trials
            ):
                # We've reached the maximum number of trials
                return False

            # Select a random target from the context
            target = self.get_next_target()

            # Create a new trial
            trial = Trial(target=target)
            self.game.trials.append(trial)

            # Notify listeners
            if self.on_trial_created:
                self.on_trial_created(trial)

            return True
        return True

    def generate_message_for_current_trial(self):
        """Generate a message for the current trial (dummy implementation)"""
        trial = self.get_current_trial()
        if trial and trial.message is None:
            message = self.agent.generate(self.game)
            trial.message = message

            if self.on_message_generated:
                self.on_message_generated(trial.message)
            return trial.message
        return None

    def make_selection(self, selection: str):
        """Make a selection for the current trial"""
        trial = self.get_current_trial()
        if trial and trial.selection is None:
            trial.selection = selection

            # Check if correct
            correct = selection == trial.target
            trial.correct = correct

            # Notify listener
            if self.on_selection_made:
                self.on_selection_made(selection, correct)

            # Move to the next trial
            self.complete_current_trial()
            return True
        return False

    def complete_current_trial(self):
        """Complete the current trial and move to the next"""
        # Move to next trial
        self.current_trial_idx += 1

        # Check if we completed the game
        if self.is_game_completed():
            if self.on_game_completed:
                self.on_game_completed()
        else:
            if self.on_trial_completed:
                self.on_trial_completed()

            # Ensure the next trial exists
            self._ensure_current_trial_exists()

    def get_progress(self):
        """Get progress information about the game"""
        completed_trials = sum(
            1 for trial in self.game.trials if trial.selection is not None
        )

        # Calculate total trials based on max_trials if set
        total_trials = self.max_trials if self.max_trials is not None else "∞"

        return {
            "current_trial": self.current_trial_idx + 1,
            "total_trials": total_trials,
            "completed_trials": completed_trials,
        }

    def get_game_summary(self):
        """Get a summary of the game results"""
        completed_trials = [
            trial for trial in self.game.trials if trial.selection is not None
        ]
        correct_count = sum(1 for trial in completed_trials if trial.correct)
        total_trials = len(completed_trials)

        # Handle division by zero
        accuracy = (correct_count / total_trials * 100) if total_trials > 0 else 0

        # Analyze by target
        target_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for trial in completed_trials:
            target_stats[trial.target]["total"] += 1
            if trial.correct:
                target_stats[trial.target]["correct"] += 1

        # Calculate accuracy by target
        target_accuracy = {}
        for target, stats in target_stats.items():
            if stats["total"] > 0:
                target_accuracy[target] = stats["correct"] / stats["total"] * 100
            else:
                target_accuracy[target] = 0

        return {
            "total_trials": total_trials,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "target_stats": dict(target_stats),
            "target_accuracy": target_accuracy,
        }


class ReferenceGameUI:
    """
    User interface for the Repeated Reference Game.
    Handles the display and interaction with the game.
    """

    def __init__(self, game_manager: GameManager):
        self.game_manager = game_manager
        self.game = game_manager.game

        # Set up callbacks
        self.game_manager.on_message_generated = self.on_message_generated
        self.game_manager.on_selection_made = self.on_selection_made
        self.game_manager.on_trial_completed = self.on_trial_completed
        self.game_manager.on_game_completed = self.on_game_completed

        # Create UI components
        self.create_ui_components()

        # Initialize UI
        self.update_ui()

    def create_ui_components(self):
        """Create all UI components"""
        # Title
        self.title = widgets.HTML(value="<h2>Repeated Reference Game</h2>")

        # Context display
        self.context_label = widgets.HTML(value="<b>Context Items:</b>")
        self.context_display = widgets.HBox([])

        # Current message display
        self.message_label = widgets.HTML(value="<b>Current Message:</b>")
        self.message_display = widgets.HTML(value="")

        # Selection buttons
        self.selection_label = widgets.HTML(value="<b>Make Selection:</b>")
        self.selection_buttons = widgets.GridBox(
            children=[],
            layout=widgets.Layout(
                grid_template_columns="repeat(5, 150px)", grid_gap="10px"
            ),
        )

        # Status display
        self.status_display = widgets.HTML(value="")

        # Progress display
        self.progress_label = widgets.HTML(value="<b>Progress:</b>")
        self.progress_display = widgets.HTML(value="")

        # Game history
        self.history_label = widgets.HTML(value="<b>Game History:</b>")
        self.history_display = widgets.HTML(value="")

        # Layout all components
        self.container = widgets.VBox(
            [
                self.title,
                widgets.HBox([self.context_label]),
                self.context_display,
                widgets.HBox([self.message_label]),
                self.message_display,
                self.progress_label,
                self.progress_display,
                self.selection_label,
                self.selection_buttons,
                self.status_display,
                self.history_label,
                self.history_display,
            ]
        )

    def get_image_widget(self, image_path, width=150):
        """Create an image widget from an image path"""
        try:
            # Check if the file exists
            if os.path.exists(image_path):
                return widgets.Image(
                    value=open(image_path, "rb").read(),
                    format=os.path.splitext(image_path)[1][
                        1:
                    ],  # Get extension without the dot
                    width=width,
                )
            else:
                # File doesn't exist, create a placeholder
                return widgets.HTML(
                    f"<div style='width:{width}px;height:{width}px;background:#eee;display:flex;justify-content:center;align-items:center;'>Image not found:<br>{image_path}</div>"
                )
        except Exception as e:
            # Handle any errors
            return widgets.HTML(
                f"<div style='width:{width}px;height:{width}px;background:#ffeeee;display:flex;justify-content:center;align-items:center;'>Error loading image:<br>{str(e)}</div>"
            )

    def update_context_display(self):
        """Update the context display with image widgets"""
        children = []
        for item in self.game.context:
            img_widget = self.get_image_widget(item)
            # Use just the filename without path for display
            filename = os.path.basename(item)
            label = widgets.HTML(f"<div style='text-align:center'>{filename}</div>")
            box = widgets.VBox([img_widget, label])
            children.append(box)

        self.context_display.children = tuple(children)

    def update_message_display(self):
        """Update the message display with the current message"""
        trial = self.game_manager.get_current_trial()
        if trial and trial.message:
            self.message_display.value = f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>{trial.message}</div>"
        else:
            self.message_display.value = "<div>No message available</div>"

    def update_selection_buttons(self):
        """Update the selection buttons with image options"""
        # Clear previous buttons
        self.selection_buttons.children = tuple()

        if self.game_manager.is_game_completed():
            return

        # Create shuffled buttons for current context
        shuffled_context = list(self.game.context)
        random.shuffle(shuffled_context)

        buttons = []
        for item in shuffled_context:
            img_widget = self.get_image_widget(item)
            btn = widgets.Button(
                description="Select",
                layout=widgets.Layout(width="auto", height="auto"),
                tooltip=item,
            )
            btn.item = item  # Store the item as an attribute
            btn.on_click(self.on_selection_button_click)

            # Use just the filename without path for display
            filename = os.path.basename(item)
            label = widgets.HTML(f"<div style='text-align:center'>{filename}</div>")
            box = widgets.VBox([img_widget, btn, label])
            buttons.append(box)

        self.selection_buttons.children = tuple(buttons)

    def update_progress_display(self):
        """Update the progress display"""
        progress = self.game_manager.get_progress()

        if progress["total_trials"] == "∞":
            # Infinite trials
            self.progress_display.value = (
                f"Trial: {progress['current_trial']} (unlimited) | "
                f"Completed: {progress['completed_trials']}"
            )
        else:
            # Fixed number of trials
            self.progress_display.value = (
                f"Trial: {progress['current_trial']}/{progress['total_trials']} | "
                f"Completed: {progress['completed_trials']}/{progress['total_trials']}"
            )

    def update_history_display(self):
        """Update the history display"""
        history = ""
        for i, trial in enumerate(self.game.trials):
            if trial.selection is not None:
                correct_mark = "✓" if trial.correct else "✗"
                target_name = os.path.basename(trial.target)
                selection_name = os.path.basename(trial.selection)
                feedback = (
                    f"Intended referent: {target_name} |"
                    if self.game_manager.feedback_label
                    else ""
                )

                history += (
                    f"<div style='margin-bottom: 5px; padding: 5px; "
                    f"background-color: {'#e6ffe6' if trial.correct else '#ffe6e6'};'>"
                    f"<b>Trial {i+1}:</b> | "
                    f"Message: {trial.message} | "
                    f"Selection: {selection_name} | {feedback}"
                    f"{correct_mark}"
                    f"</div>"
                )
        self.history_display.value = history

    def update_status_display(self, message=""):
        """Update the status display"""
        self.status_display.value = f"<p>{message}</p>"

    def update_ui(self):
        """Update all UI components"""
        self.update_context_display()
        self.update_message_display()
        self.update_progress_display()
        self.update_selection_buttons()
        self.update_history_display()

    def show_game_summary(self):
        """Display a summary of the game results"""
        summary = self.game_manager.get_game_summary()

        html_summary = (
            f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
            f"<h3>Game Summary</h3>"
            f"<p>Total Trials: {summary['total_trials']}</p>"
            f"<p>Correct Selections: {summary['correct_count']}</p>"
            f"<p>Accuracy: {summary['accuracy']:.1f}%</p>"
            f"<h4>Accuracy by Target:</h4>"
            f"<ul>"
        )

        for target, accuracy in summary["target_accuracy"].items():
            target_name = os.path.basename(target)
            stats = summary["target_stats"][target]
            html_summary += f"<li>{target_name}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})</li>"

        html_summary += f"</ul>" f"</div>"

        self.status_display.value = html_summary

    # Event handlers
    def on_selection_button_click(self, btn):
        """Handle selection button click"""
        selection = btn.item
        self.game_manager.make_selection(selection)

    # Callback handlers from GameManager
    def on_message_generated(self, message):
        """Called when a message is generated"""
        self.update_message_display()

    def on_selection_made(self, selection, correct):
        """Called when a selection is made"""
        selection_name = os.path.basename(selection)
        trial = self.game_manager.get_current_trial()

        self.update_status_display(
            f"{'Correct!' if correct else 'Incorrect!'} "
            f"You selected '{selection_name}'."
        )

    def on_trial_completed(self):
        """Called when a trial is completed"""
        # Generate message for the next trial
        with open(self.game_manager.game_save_path, "w") as f:
            json.dump(self.game_manager.game.model_dump(mode="json"), f)
        self.game_manager.generate_message_for_current_trial()
        self.update_ui()

    def on_game_completed(self):
        """Called when the game is completed"""
        self.selection_buttons.children = tuple()
        self.update_ui()
        self.update_status_display("<h3>Game Complete!</h3>")
        self.show_game_summary()
        with open(self.game_manager.game_save_path, "w") as f:
            json.dump(self.game_manager.game.model_dump(mode="json"), f)

    def display(self):
        """Display the UI"""
        display(self.container)
        return self.container


def initialize_game(image_paths):
    """
    Initialize a new RepeatedReferenceGame with the given image paths.
    Does not pre-generate trials, allowing for dynamic trial generation.

    Args:
        image_paths: Tuple or list of image paths

    Returns:
        RepeatedReferenceGame instance
    """
    # Convert to tuple if list is provided
    context = tuple(image_paths) if isinstance(image_paths, list) else image_paths

    # Create the game with empty trials list
    game = RepeatedReferenceGame(context=context)

    return game


def run_reference_game(
    agent, image_paths, game_save_path, max_trials=None, next_target_policy="block"
):
    """
    Run the reference game with the specified image paths

    Args:
        image_paths: List of image file paths to use in the game
        max_trials: Maximum number of trials to run (None for unlimited)

    Returns:
        Tuple of (game, game_manager, ui)
    """
    game = initialize_game(image_paths)

    # Create game manager and UI
    game_manager = GameManager(
        agent,
        game,
        game_save_path,
        max_trials=max_trials,
        next_target_policy=next_target_policy,
    )
    ui = ReferenceGameUI(game_manager)

    # This will create the first trial and generate a message for it
    game_manager._ensure_current_trial_exists()
    game_manager.generate_message_for_current_trial()

    return game, game_manager, ui


def run_sequential_games(
    agent, image_paths_list, game_save_paths_list, max_trials_list=None
):
    """
    Run multiple reference games in sequence

    Args:
        image_paths_list: A list of lists, where each inner list contains image paths for one game
        max_trials_list: A list of max_trials values for each game (None for unlimited)

    Returns:
        A list of (game, game_manager, ui) tuples for each game
    """
    if max_trials_list is None:
        max_trials_list = [None] * len(image_paths_list)

    # Create all games but don't display them yet
    games_data = []
    ui_containers = []

    for i, (image_paths, save_path, max_trials) in enumerate(
        zip(image_paths_list, game_save_paths_list, max_trials_list)
    ):
        # Initialize the game
        game = initialize_game(image_paths)
        game_manager = GameManager(agent, game, save_path, max_trials=max_trials)
        ui = ReferenceGameUI(game_manager)

        # Create container to hold the UI with a visible flag
        container = widgets.VBox([ui.container], layout=widgets.Layout(display="none"))
        if i == 0:
            # Show only the first game initially
            container.layout.display = "flex"

        ui_containers.append(container)
        games_data.append((game, game_manager, ui))

        # When a game completes, move to the next one
        if i < len(image_paths_list) - 1:
            # Not the last game, set up transition to next game
            def create_completion_handler(current_idx):
                def on_game_completed():
                    # Hide current game
                    ui_containers[current_idx].layout.display = "none"
                    # Show next game
                    ui_containers[current_idx + 1].layout.display = "flex"
                    # Initialize the next game
                    next_game = games_data[current_idx + 1]
                    next_game[1]._ensure_current_trial_exists()
                    next_game[1].generate_message_for_current_trial()
                    with open(game_manager.game_save_path, "w") as f:
                        json.dump(game_manager.game.model_dump(mode="json"), f)

                return on_game_completed

            game_manager.on_game_completed = create_completion_handler(i)

    # Initialize first game
    first_game = games_data[0]
    first_game[1]._ensure_current_trial_exists()
    first_game[1].generate_message_for_current_trial()

    # Display all containers
    container = widgets.VBox(ui_containers)
    display(container)

    return games_data


# Example usage:
# All games will be loaded at once, but only displayed sequentially
# image_paths_game1 = [...] # first set of images
# image_paths_game2 = [...] # second set of images
# image_paths_game3 = [...] # third set of images
#
# games = run_sequential_games(
#     [image_paths_game1, image_paths_game2, image_paths_game3],
#     [10, 8, 12]  # Each game has a different number of trials
# )
