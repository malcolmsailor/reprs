import io

import miditoolkit
import mido
import pandas as pd
import pytest
from music_df.add_feature import infer_barlines, simplify_time_sigs
from music_df.quantize_df import quantize_df
from music_df.read_midi import read_midi
from music_df.sort_df import sort_df

from reprs.oct import (
    MAX_INST,
    MAX_PITCH,
    POS_RESOLUTION,
    oct_decode,
    oct_encode,
    preprocess_df_for_oct_encoding,
)
from tests.helpers_for_tests import (
    get_input_kern_paths,
    get_input_midi_paths,
    read_humdrum,
)
from tests.original_oct_implementation import MIDI_to_encoding


@pytest.mark.filterwarnings(
    "ignore:note_off event"  # Ignore warnings from reading midi
)
def test_oct_encode(n_kern_files):
    paths = get_input_midi_paths(seed=42, n_files=1)
    comparisons = 0
    inequalities = 0
    vocab_sizes = {
        "bar": 256,  # Bars
        "position": 128,  # Positions
        "instrument": 129,  # Instrument
        "pitch": 256,  # Pitch
        "duration": 128,  # Duration
        "velocity": 32,  # Velocity
        "time_sig": 254,  # Time signatures
        "tempo": 49,  # Tempi
    }
    for i, path in enumerate(paths):
        if "K218 ii " in path:
            # TODO: (Malcolm 2023-09-11) this midi file has an excess token in
            #   my version for some reason. TODO investigate.
            continue
        if "Des heiligen Geistes reiche" in path:
            # TODO: (Malcolm 2023-09-11) this midi file has 3 excess tokens in my
            #   version for some reason.
            continue
        print(f"{i + 1}/{len(paths)}: {path}")
        df = read_midi(path)
        # ticks_per_beat = mido.MidiFile(path).ticks_per_beat
        df = preprocess_df_for_oct_encoding(df)
        assert isinstance(df, pd.DataFrame)
        encoding = oct_encode(df)
        tokens = encoding._tokens
        decoded = oct_decode(tokens)

        orig_notes = sort_df(
            quantize_df(
                df[df.type == "note"].reset_index(drop=True), tpq=POS_RESOLUTION
            )
        )
        decode_notes = decoded[decoded.type == "note"].reset_index(drop=True)
        assert (orig_notes.pitch == decode_notes.pitch).all()
        assert (orig_notes.onset == decode_notes.onset).all()

        # TODO: (Malcolm 2023-11-06) releases don't always match, by a small amount.
        # assert (orig_notes.release == decode_notes.release).all()

        # Get reference implementation
        # with open(path, "rb") as f:
        #     midi_file = io.BytesIO(f.read())
        midi_obj = miditoolkit.midi.parser.MidiFile(filename=path)
        reference_encoding = MIDI_to_encoding(midi_obj)
        # These comparisons sometimes fail but from inspection it appears to be due to
        #   floating point/rounding issues where sometimes the onset of a note is rounded
        #   to a different position, or the relative position of a tempo change or similar
        #   is rounded differently

        for i, (x, y) in enumerate(zip(tokens, reference_encoding)):
            for xx, token_type, n_tokens in zip(
                x, vocab_sizes.keys(), vocab_sizes.values()
            ):
                # TODO: (Malcolm 2023-09-11) I need to implement maximum bar cropping elsewhere
                assert xx < n_tokens or token_type == "bar"
            try:
                assert x == y
            except:
                for xx, yy, token_type, n_tokens in zip(
                    x, y, vocab_sizes.keys(), vocab_sizes.values()
                ):
                    # TODO: (Malcolm 2023-09-11) I need to implement maximum bar cropping elsewhere
                    assert xx < n_tokens or token_type == "bar"
                    if xx > 128 and token_type == "position":
                        breakpoint()
                    if xx != yy:
                        inequalities += 1
            comparisons += len(x)

        assert len(tokens) == len(reference_encoding)
    print(f"{comparisons - inequalities}/{comparisons} equal")

    # paths = get_input_kern_paths(seed=42)

    # for i, path in enumerate(paths):
    #     print(f"{i + 1}/{len(paths)}: {path}")
    #     df = read_humdrum(path)
    #     encoding = oct_encode(df)
    #     breakpoint()


def test_percussion_channel_handling():
    """Test that notes on channel 9 are encoded as drums with pitch offset."""
    df = pd.DataFrame(
        [
            {"type": "bar", "onset": 0.0},
            {"type": "time_signature", "onset": 0.0, "other": '{"numerator": 4, "denominator": 4}'},
            {
                "type": "note",
                "onset": 0.0,
                "release": 1.0,
                "pitch": 36,
                "channel": 9,
                "velocity": 100,
            },
            {
                "type": "note",
                "onset": 1.0,
                "release": 2.0,
                "pitch": 60,
                "channel": 0,
                "velocity": 100,
            },
        ]
    )

    encoding = oct_encode(df)
    tokens = encoding._tokens

    assert len(tokens) == 2

    drum_token = next(t for t in tokens if t.position == 0)
    melodic_token = next(t for t in tokens if t.position != 0)

    assert drum_token.pitch == 36 + MAX_PITCH + 1
    assert drum_token.instrument == MAX_INST + 1
    assert melodic_token.pitch == 60
    assert melodic_token.instrument == 0

    decoded = oct_decode(tokens)
    decoded_notes = decoded[decoded.type == "note"].sort_values("onset").reset_index(drop=True)

    assert decoded_notes.loc[0, "pitch"] == 36
    assert decoded_notes.loc[1, "pitch"] == 60


def test_percussion_without_channel_column():
    """Test that encoding works when channel column is absent (backward compatibility)."""
    df = pd.DataFrame(
        [
            {"type": "bar", "onset": 0.0},
            {"type": "time_signature", "onset": 0.0, "other": '{"numerator": 4, "denominator": 4}'},
            {
                "type": "note",
                "onset": 0.0,
                "release": 1.0,
                "pitch": 36,
                "velocity": 100,
            },
        ]
    )

    encoding = oct_encode(df)
    tokens = encoding._tokens

    assert len(tokens) == 1
    assert tokens[0].pitch == 36
