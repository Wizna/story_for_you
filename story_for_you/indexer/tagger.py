from story_for_you.analysis.context import StoryContext
from story_for_you.indexer.segment import Segment


class CharacterTagger:
    """Annotates segments with character participation."""

    def __init__(self, context: StoryContext):
        self.context = context

    def tag(self, segments: list[Segment]) -> list[Segment]:
        """Populate segment characters using context knowledge."""
        raise NotImplementedError
