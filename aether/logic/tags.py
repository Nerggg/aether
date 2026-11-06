"""Parser for the state tags the DM can write, e.g. [TAKE_DAMAGE: 3].

The old interceptor treated every "[" as a tag start, so a stray bracket
swallowed text until the next "]". This parser only holds text back while it
could still become a *known* tag, and gives it back as normal text the moment
that is no longer possible.

Streaming use:
    parser = TagParser()
    for chunk in llm_stream:
        result = parser.feed(chunk)
        show(result.text)
        for tag in result.tags: apply(tag)
    result = parser.flush()      # always call at the end of the stream
    show(result.text)

Whole-text use:
    clean_text, tags = parse_tags(text)

Rolls are not tags any more: the option's skill tag or the judge decides when
to roll. Pass `names=` if you want a different set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

DEFAULT_TAGS = ("ADD_ITEM", "REMOVE_ITEM", "TAKE_DAMAGE", "HEAL_HP")
MAX_TAG_LENGTH = 80


@dataclass(frozen=True)
class Tag:
    name: str   # always upper case
    value: str


@dataclass
class ParseResult:
    text: str = ""
    tags: List[Tag] = field(default_factory=list)


class TagParser:
    def __init__(self, names: Sequence[str] = DEFAULT_TAGS, max_length: int = MAX_TAG_LENGTH):
        self.names = tuple(n.upper() for n in names)
        self.max_length = max_length
        self._buf = ""  # non-empty while a "[" might still turn into a tag
        alternatives = "|".join(re.escape(n) for n in self.names)
        self._pattern = re.compile(rf"^\[\s*({alternatives})\s*:\s*([^\[\]]*?)\s*\]$", re.IGNORECASE)

    def _could_be_tag(self, buf: str) -> bool:
        if len(buf) > self.max_length or "\n" in buf:
            return False
        body = buf[1:]
        if ":" in body:
            return body.split(":", 1)[0].strip().upper() in self.names
        upper = body.strip().upper()
        return any(name.startswith(upper) for name in self.names)

    def _match(self, buf: str) -> Optional[Tag]:
        m = self._pattern.match(buf)
        if not m or not m.group(2):
            return None  # unknown name or empty value: leave it as plain text
        return Tag(m.group(1).upper(), m.group(2))

    def feed(self, chunk: str) -> ParseResult:
        out: List[str] = []
        tags: List[Tag] = []
        for ch in chunk:
            if self._buf:
                if ch == "[":
                    # The held text was not a tag; this bracket may start a new one.
                    out.append(self._buf)
                    self._buf = "["
                    continue
                self._buf += ch
                if ch == "]":
                    tag = self._match(self._buf)
                    if tag:
                        tags.append(tag)
                    else:
                        out.append(self._buf)
                    self._buf = ""
                elif not self._could_be_tag(self._buf):
                    out.append(self._buf)
                    self._buf = ""
            elif ch == "[":
                self._buf = "["
            else:
                out.append(ch)
        return ParseResult("".join(out), tags)

    def flush(self) -> ParseResult:
        """End of stream: release anything still held back as plain text."""
        text, self._buf = self._buf, ""
        return ParseResult(text, [])


def parse_tags(text: str, names: Sequence[str] = DEFAULT_TAGS) -> Tuple[str, List[Tag]]:
    """Split finished text into (text without tags, tags found)."""
    parser = TagParser(names)
    first = parser.feed(text)
    last = parser.flush()
    return (first.text + last.text).rstrip(), first.tags


def parse_amount(value: str) -> Optional[int]:
    """First whole number in a tag value ('3', '3 damage', '-2'), or None."""
    m = re.search(r"-?\d+", value or "")
    return int(m.group()) if m else None
