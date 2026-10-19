"""The Mechanist mark: three cells in a loop; the heavy blue arrow is the step the evidence points to.

FULL is drawn for 26 px and up, SMALL for 16-20 px (the dotted arc drops out, strokes thicken). Both take the ink from
currentColor and the blue from var(--logo) unless a literal is passed.
"""
from urllib.parse import quote

INK, BLUE, INK_DARK, BLUE_DARK = "#161513", "#2b6cd4", "#f1eee6", "#8db4ec"


def _full(blue):
    return (f'<circle cx="32" cy="12" r="7.5" fill="none" stroke="currentColor" stroke-width="3.4"/>'
            f'<circle cx="13" cy="47" r="7.5" fill="none" stroke="currentColor" stroke-width="3.4"/>'
            f'<circle cx="51" cy="47" r="7.5" fill="currentColor"/>'
            f'<path d="M38.8 17 Q50.5 24.5 50.8 36" fill="none" stroke="{blue}" stroke-width="4.6" stroke-linecap="round"/>'
            f'<path d="M46.2 31.4 L50.9 37.4 L55.2 31.2" fill="none" stroke="{blue}" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M43 50.5 Q32 56.5 21.8 50.6" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>'
            f'<path d="M25.4 47.2 L21 50.4 L25.6 53.6" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M11.6 38.8 Q13 27 24.4 17.4" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-dasharray="0.1 5.4"/>')


def _small(blue):
    return (f'<circle cx="32" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="6"/>'
            f'<circle cx="12" cy="49" r="9" fill="none" stroke="currentColor" stroke-width="6"/>'
            f'<circle cx="52" cy="49" r="10" fill="currentColor"/>'
            f'<path d="M41 17.5 Q51.5 25 52 33.5" fill="none" stroke="{blue}" stroke-width="7" stroke-linecap="round"/>'
            f'<path d="M46 29 L52 35.5 L57.5 29" fill="none" stroke="{blue}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M41.5 54 Q32 59 22.5 54" fill="none" stroke="currentColor" stroke-width="4.5" stroke-linecap="round"/>')


def svg(small=False, blue="var(--logo)", cls="mk"):
    body = _small(blue) if small else _full(blue)
    return f'<svg class="{cls}" viewBox="0 0 64 64" aria-hidden="true" focusable="false">{body}</svg>'


def favicon_link():
    """A tab icon that follows the browser's light or dark setting."""
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><style>svg{{color:{INK}}}.b{{stroke:{BLUE}}}'
         f'@media (prefers-color-scheme:dark){{svg{{color:{INK_DARK}}}.b{{stroke:{BLUE_DARK}}}}}</style>'
         + _small("currentColor").replace(f'stroke="currentColor" stroke-width="7"', 'class="b" stroke-width="7"') + '</svg>')
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,{quote(s)}">'
