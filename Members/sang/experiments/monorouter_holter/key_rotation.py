"""Sequential MonoRouter key fallback for the single-threaded Holter notebook."""

from urllib.error import HTTPError


class KeyRotator:
    def __init__(self, primary, secondary="", on_switch=None):
        self._keys = []
        for key in (primary, secondary):
            if key and key not in self._keys:
                self._keys.append(key)
        self._index = 0
        self._on_switch = on_switch

    def execute(self, operation):
        if not self._keys:
            raise RuntimeError("MONOROUTER_API_KEY를 .env 또는 환경변수에 설정하세요.")
        while True:
            try:
                return operation(self._keys[self._index])
            except HTTPError as exc:
                # HTTP 402 can indicate key allowance or shared account balance.
                # Do not switch on authentication, authorization or rate-limit errors.
                if exc.code != 402 or self._index + 1 >= len(self._keys):
                    raise
                self._index += 1
                if self._on_switch is not None:
                    self._on_switch(self._index + 1)
