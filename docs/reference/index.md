# Reference

Look-up material for `mudm-tools`. For step-by-step walkthroughs, see the
[Guides](../guides/index.md); this section is the precise specification and
signature surface.

| Page | Contents |
| --- | --- |
| [TileJSON for muDM](tilejson.md) | The tileset-metadata specification (TileJSON 3.0.0 + muDM extensions: multiscale, axes, units), worked JSON examples, and the `mudm.tilemodel` Pydantic models. |
| [Command-Line Reference](cli.md) | The `mudm-serve` viewer server and the `python -m mudm_tools.converters.cli` converter CLI — every flag, plus what each server route serves. |
| [Python API](python-api.md) | The public `mudm_tools` API map: top-level exports, the Rust tiling engines (`mudm_tools._rs`), and links to the in-context autodoc in each guide. |

!!! info "Where the full autodoc lives"
    To keep each symbol documented in exactly one place, the rendered
    signatures, parameters, and return types appear **in the guide** that uses
    them. The [Python API](python-api.md) page is the index that routes you
    there, and hand-documents the compiled Rust engines that autodoc cannot
    introspect.
