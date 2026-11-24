from transformers import PretrainedConfig


class SophonRealtimeConfig(PretrainedConfig):
    """Root config: only describes the layout (thinker/, talker/, voices/ subdirs); each stage's weights live in its own subdir."""
    model_type = "sophon_realtime"

    def __init__(self, thinker_subdir="thinker", talker_subdir="talker", voices_subdir="voices", default_voice="chelsie", default_language="zh", **kw):
        self.thinker_subdir = thinker_subdir; self.talker_subdir = talker_subdir; self.voices_subdir = voices_subdir
        self.default_voice = default_voice; self.default_language = default_language
        super().__init__(**kw)


class SophonE2EStageConfig(PretrainedConfig):
    """Config of the talker220/ folder: the talker and decoder stages are attention-free shells around the E2E engine."""
    model_type = "sophon_e2e_tts"

    def __init__(self, vocab_size=151936, hidden_size=16, max_position_embeddings=8192, **kw):
        self.vocab_size = vocab_size; self.hidden_size = hidden_size; self.max_position_embeddings = max_position_embeddings
        super().__init__(**kw)
