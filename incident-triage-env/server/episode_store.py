"""Module-level store for completed episode results.

The environment writes to this on episode end; the /grader endpoint reads from it.
This bridges the gap since create_app() manages environment lifecycle internally.
"""

_completed_episodes: dict[str, dict] = {}


def store_episode(episode_id: str, data: dict) -> None:
    _completed_episodes[episode_id] = data


def get_episode(episode_id: str) -> dict | None:
    return _completed_episodes.get(episode_id)


def list_episodes() -> list[str]:
    return list(_completed_episodes.keys())
