"""Tests for capitalization.py (sentence-case post-processing for NAR models)."""

from faster_whisper_hotkey.capitalization import add_capitalization


class TestAddCapitalization:
    def test_empty_string(self):
        assert add_capitalization("") == ""

    def test_no_letters(self):
        assert add_capitalization("123 456") == "123 456"

    def test_first_word_capitalized(self):
        assert add_capitalization("hello world") == "Hello world"

    def test_leading_punctuation_skipped(self):
        assert add_capitalization("  (hello) world") == "  (Hello) world"

    def test_period_starts_new_sentence(self):
        assert add_capitalization("hello. how are you") == "Hello. How are you"

    def test_question_mark_and_exclamation_mark(self):
        assert add_capitalization("really? yes! no way") == "Really? Yes! No way"

    def test_decimal_point_does_not_start_sentence(self):
        assert add_capitalization("pi is 3.14. it is irrational") == "Pi is 3.14. It is irrational"

    def test_closing_quote_before_sentence_start(self):
        assert add_capitalization('he said "hi". then he left') == 'He said "hi". Then he left'

    def test_standalone_i_english(self):
        assert add_capitalization("i was there. i saw it") == "I was there. I saw it"

    def test_i_contractions_english(self):
        assert add_capitalization("i'm fine, i'll be there") == "I'm fine, I'll be there"

    def test_i_not_touched_inside_words(self):
        assert add_capitalization("the wifi is flaky") == "The wifi is flaky"

    def test_standalone_i_only_for_english(self):
        assert add_capitalization("du und i sind da", "de") == "Du und i sind da"
        assert add_capitalization("du und i sind da", "en") == "Du und I sind da"

    def test_cjk_unchanged(self):
        text = "こんにちは。元気ですか？"
        assert add_capitalization(text, "ja") == text

    def test_already_capitalized_untouched(self):
        assert add_capitalization("Hello world. How are you?") == "Hello world. How are you?"

    def test_idempotent(self):
        text = "so yeah, i was wondering what did you do? and then what?"
        once = add_capitalization(text)
        assert add_capitalization(once) == once

    def test_german_esh_at_sentence_start(self):
        assert add_capitalization("die sonne scheint. straße 5 hier", "de") == ("Die sonne scheint. Straße 5 hier")

    def test_user_example(self):
        text = (
            "so yeah, i was wondering what did you do yesterday exactly? "
            "and when you went to the cinema afterward was the film good? "
            "about that do you think mary liked it?"
        )
        expected = (
            "So yeah, I was wondering what did you do yesterday exactly? "
            "And when you went to the cinema afterward was the film good? "
            "About that do you think mary liked it?"
        )
        assert add_capitalization(text) == expected
