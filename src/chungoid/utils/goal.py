"""Utilities for parsing and validating the goal.txt file."""

import logging
import re
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_goal_file(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Parses the goal.txt file to extract goal and environment descriptions.

    Expects the file to contain lines starting with 'goal:' and 'env:'.

    Args:
        file_path: The path to the goal.txt file.

    Returns:
        A tuple containing (goal_description, environment_description).
        Returns (None, None) if the file cannot be read or doesn't contain
        the expected keys.
    """
    goal: Optional[str] = None
    env: Optional[str] = None
    logger.info("Parsing goal file: %s", file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.lower().startswith("goal:"):
                # Take everything after 'goal:' and strip leading/trailing whitespace
                goal_content = stripped_line[len("goal:") :].strip()
                if goal_content:
                    goal = goal_content
                    logger.debug("Found goal description.")
            elif stripped_line.lower().startswith("env:"):
                # Take everything after 'env:' and strip leading/trailing whitespace
                env_content = stripped_line[len("env:") :].strip()
                if env_content:
                    env = env_content
                    logger.debug("Found environment description.")

        if not goal:
            logger.warning("'goal:' key not found or empty in %s", file_path)
        if not env:
            logger.warning("'env:' key not found or empty in %s", file_path)

        return goal, env

    except FileNotFoundError:
        logger.error("Goal file not found: %s", file_path)
        return None, None
    except IOError as e:
        logger.error("Error reading goal file %s: %s", file_path, e)
        return None, None
    except Exception as e:
        logger.exception("Unexpected error parsing goal file %s: %s", file_path, e)
        return None, None


def validate_goal_content(goal: Optional[str], env: Optional[str]) -> Dict[str, bool]:
    """Performs basic validation checks on the parsed goal and environment.

    Args:
        goal: The parsed goal description string.
        env: The parsed environment description string.

    Returns:
        A dictionary indicating validation results (e.g., {'goal_present': True, 'env_present': False}).
    """
    results = {
        "goal_present": bool(goal and goal.strip()),
        "env_present": bool(env and env.strip()),
        # Add more specific checks as needed
        # Example: Check for keywords, length constraints, etc.
        "goal_plausible_length": len(goal) > 10 if goal else False,
        "env_mentions_language": (
            bool(re.search(r"python|javascript|java|c#|go|rust", env.lower())) if env else False
        ),
    }
    logger.info("Goal content validation results: %s", results)
    return results


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create a dummy goal.txt for testing
    dummy_file = "./temp_goal.txt"
    content = """
    goal: Create a web server using Python and Flask to serve a simple API.
            It should have endpoints for users and items.

    env: Target environment is Linux (Ubuntu 22.04), Python 3.10+.
         Requires requests and Flask libraries.
         Deployment via Docker.
    """
    try:
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Test parsing
        parsed_goal, parsed_env = parse_goal_file(dummy_file)
        print(f"\nParsed Goal: {parsed_goal}")
        print(f"Parsed Env: {parsed_env}")

        # Test validation
        if parsed_goal and parsed_env:
            validation = validate_goal_content(parsed_goal, parsed_env)
            print(f"\nValidation Results: {validation}")

        # Test file not found
        print("\nTesting non-existent file:")
        parse_goal_file("./non_existent_goal.txt")

    finally:
        # Clean up dummy file
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            print(f"\nCleaned up {dummy_file}")
