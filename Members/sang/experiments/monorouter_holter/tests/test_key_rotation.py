import unittest
from urllib.error import HTTPError

from Members.sang.experiments.monorouter_holter.key_rotation import KeyRotator


def error(code):
    return HTTPError("https://example.invalid", code, "error", {}, None)


class KeyRotationTest(unittest.TestCase):
    def test_first_key_used_when_successful(self):
        keys = []
        rotator = KeyRotator("first", "second")
        self.assertEqual(rotator.execute(lambda key: keys.append(key) or "ok"), "ok")
        self.assertEqual(keys, ["first"])

    def test_402_switches_once_and_stays_on_second(self):
        keys, switches = [], []
        rotator = KeyRotator("first", "second", switches.append)

        def operation(key):
            keys.append(key)
            if key == "first":
                raise error(402)
            return "ok"

        self.assertEqual(rotator.execute(operation), "ok")
        self.assertEqual(rotator.execute(operation), "ok")
        self.assertEqual(keys, ["first", "second", "second"])
        self.assertEqual(switches, [2])

    def test_other_errors_do_not_switch(self):
        for code in (401, 403, 429, 500):
            keys = []

            def operation(key):
                keys.append(key)  # noqa: B023
                raise error(code)  # noqa: B023

            with self.assertRaises(HTTPError):
                KeyRotator("first", "second").execute(operation)
            self.assertEqual(keys, ["first"])

    def test_both_exhausted_propagates_402(self):
        keys = []

        def operation(key):
            keys.append(key)
            raise error(402)

        with self.assertRaises(HTTPError) as caught:
            KeyRotator("first", "second").execute(operation)
        self.assertEqual(caught.exception.code, 402)
        self.assertEqual(keys, ["first", "second"])

    def test_same_key_is_not_retried_and_no_key_errors(self):
        keys = []

        def operation(key):
            keys.append(key)
            raise error(402)

        with self.assertRaises(HTTPError):
            KeyRotator("same", "same").execute(operation)
        self.assertEqual(keys, ["same"])
        with self.assertRaises(RuntimeError):
            KeyRotator("").execute(operation)


if __name__ == "__main__":
    unittest.main()
