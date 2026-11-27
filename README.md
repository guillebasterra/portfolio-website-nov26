# Portfolio website, Nov 2026

Static prototypes, no build step. Open `prototype/page/index.html` in a browser.

- `prototype/page/`: the page. A stage on the left (drawing, ASCII or brick head), the column on the right.
- `prototype/head/`: the brick head on its own.
- `assets/work/art/`, `assets/work/photography/`: the pictures the page shows, in filename order.
- `tools/`: scripts that generate the data files under `prototype/`. Run them from the repo root.

## Changing the pictures

Add, remove or rename files in `assets/work/art/` or `assets/work/photography/`, then:

    python3 tools/work/catalog.py

Use web-sized jpg, png or webp. The script flags anything over 2400px or 1.5 MB and skips formats browsers can't show.
