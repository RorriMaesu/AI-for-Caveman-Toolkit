# Asset Requirements

Place the following assets before real-mode generation:

1. `book_cover.png` - used as bottom-right overlay in final assembly.
2. Optional logos/branding artwork.
3. Any approved stock textures for backgrounds.

## Recommended book cover format
- PNG with transparency supported.
- Minimum 1024px width.
- Keep focal elements centered to avoid clipping when scaled.

The controller expects the cover path from `config.yaml`:
`paths.cover_image: _assets/book_cover.png`
